# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Coordinator for multi-agent incident investigation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Optional

import asyncpg

from smart_city.llm.agents.base import TraceStep
from smart_city.llm.agents.investigator import InvestigatorAgent
from smart_city.llm.agents.sop_advisor import SOPAdvisorAgent
from smart_city.llm.agents.tools import execute_tool
from smart_city.llm.vector_store import MilvusException

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """Coordinates specialist agents and synthesizes final report."""

    def __init__(self, *, vllm_url: str, model: str, api_key: str) -> None:
        """Initialize orchestrator and specialist agents."""
        cleaned_url = vllm_url.strip().rstrip("/")
        self._vllm_url = (
            cleaned_url[:-3] if cleaned_url.endswith("/v1") else cleaned_url
        )
        self._model = model
        self._api_key = api_key
        self._investigator = InvestigatorAgent(
            vllm_url=vllm_url,
            model=model,
            api_key=api_key,
        )
        self._sop_advisor = SOPAdvisorAgent(
            vllm_url=vllm_url,
            model=model,
            api_key=api_key,
        )

    async def run(
        self,
        *,
        zone_id: str,
        hours: int,
        db_pool: Optional[Any],
        vector_store: Optional[Any],
        embedder: Optional[Any],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        on_trace_update: Optional[Any] = None,
    ) -> tuple[str, list[TraceStep]]:
        """Run specialists then synthesize final incident report."""
        trace: list[TraceStep] = []

        investigator_input = (
            f"Investigate incident for zone '{zone_id}' over last {hours} hours."
        )
        if from_date and to_date:
            investigator_input = (
                f"Investigate incident for zone '{zone_id}' for date range "
                f"{from_date} to {to_date}."
            )

        trace.append(
            TraceStep(
                agent="OrchestratorAgent",
                tool=None,
                summary=(
                    f"Delegating to InvestigatorAgent: {investigator_input}"
                ),
                timestamp=datetime.now(timezone.utc),
            )
        )
        if on_trace_update:
            on_trace_update(list(trace))

        # Record the slice start so live preview steps can be replaced by the
        # official _extract_trace steps (which also include agent thoughts).
        inv_slice_start = len(trace)

        def _inv_live_step(step: "TraceStep") -> None:
            """Append a live tool step and flush to job state immediately."""
            trace.append(step)
            if on_trace_update:
                on_trace_update(list(trace))

        incident_summary, investigator_trace = await self._investigator.run(
            investigator_input,
            db_pool=db_pool,
            from_date=from_date,
            to_date=to_date,
            step_callback=_inv_live_step if on_trace_update else None,
        )
        # Replace live preview steps with the authoritative post-run trace.
        del trace[inv_slice_start:]
        trace.extend(investigator_trace)
        if on_trace_update:
            on_trace_update(list(trace))

        # Sanitize and normalize GAIA output before checking structure.
        sanitized_summary = self._normalize_investigator_output(
            self._sanitize_agent_summary(incident_summary)
        )
        needs_fallback = (
            not sanitized_summary.strip()
            or "INCIDENT_FACTS" not in sanitized_summary
        )
        if needs_fallback:
            logger.info(
                "Investigator output unparseable (INCIDENT_FACTS missing); "
                "trying to reconstruct from tool results. Raw length=%d",
                len(incident_summary),
            )
            # Prefer building from the tool results already in the trace —
            # avoids a redundant SQL round-trip and removes the fallback
            # step when the agent called all tools but the LLM's final
            # answer was unparseable (e.g. thinking tokens broke JSON).
            trace_summary = self._build_summary_from_trace(investigator_trace)
            if trace_summary:
                incident_summary = trace_summary
                needs_fallback = False
                logger.info("Reconstructed INCIDENT_FACTS from trace tool results.")

        if needs_fallback:
            logger.info(
                "No tool results in trace; running investigator tools directly."
            )
            direct_summary, direct_trace = await self._run_investigator_tools_direct(
                zone_id=zone_id,
                hours=hours,
                db_pool=db_pool,
                from_date=from_date,
                to_date=to_date,
            )
            if direct_summary:
                incident_summary = direct_summary
                trace.extend(direct_trace)
                needs_fallback = False
                if on_trace_update:
                    on_trace_update(list(trace))

        if needs_fallback:
            incident_summary = await self._fallback_investigator_summary(
                zone_id=zone_id,
                hours=hours,
                db_pool=db_pool,
                from_date=from_date,
                to_date=to_date,
            )
            trace.append(
                TraceStep(
                    agent="OrchestratorAgent",
                    tool="sql_fallback",
                    summary=(
                        "InvestigatorAgent output unparseable or empty; "
                        "SQL direct query used as fallback."
                    ),
                    timestamp=datetime.now(timezone.utc),
                )
            )

        sop_hits = self._retrieve_sop_hits(
            zone_id=zone_id,
            incident_summary=incident_summary,
            vector_store=vector_store,
            embedder=embedder,
        )
        if not sop_hits:
            raise RuntimeError(
                "No SOP policy matches found in RAG for this incident."
            )

        trace.append(
            TraceStep(
                agent="SOPAdvisorAgent",
                tool="search_sop_documents",
                summary=(
                    f"Retrieved {len(sop_hits)} SOP/policy chunks via RAG: "
                    f"{', '.join(h['source'] for h in sop_hits[:4])}"
                ),
                timestamp=datetime.now(timezone.utc),
                raw_result=json.dumps({"rows": sop_hits}),
            )
        )
        if on_trace_update:
            on_trace_update(list(trace))

        compact_incident_summary = self._compact_agent_context(
            incident_summary, max_chars=2600
        )

        sop_input = (
            "Use the incident summary below and retrieve relevant SOP guidance.\n\n"
            f"Incident summary:\n{compact_incident_summary}\n\n"
            "Query for crowd-density response, escalation, privacy safeguards, "
            "and AI operations governance. Prioritize matches from TX-SCPS-POL "
            "documents and include citations for every recommendation. "
            "If no SOP evidence is found, output NO_SOP_FOUND exactly using "
            "the fallback format from your system instructions.\n\n"
            "Retrieved policy context:\n"
            f"{self._compact_agent_context(self._format_sop_hits(sop_hits), 1800)}"
        )
        trace.append(
            TraceStep(
                agent="OrchestratorAgent",
                tool=None,
                summary=(
                    f"Delegating to SOPAdvisorAgent with {len(sop_hits)} "
                    f"policy hits for zone '{zone_id}'."
                ),
                timestamp=datetime.now(timezone.utc),
            )
        )
        if on_trace_update:
            on_trace_update(list(trace))

        sop_slice_start = len(trace)

        def _sop_live_step(step: "TraceStep") -> None:
            """Append a live SOP tool step and flush to job state."""
            trace.append(step)
            if on_trace_update:
                on_trace_update(list(trace))

        sop_summary, sop_trace = await self._sop_advisor.run(
            sop_input,
            vector_store=vector_store,
            embedder=embedder,
            step_callback=_sop_live_step if on_trace_update else None,
        )
        del trace[sop_slice_start:]
        trace.extend(sop_trace)
        # When the SOP Advisor uses the pre-provided RAG context directly
        # (no additional tool calls), emit a completion step so the UI
        # always shows the agent ran and produced a result.
        if not sop_trace:
            preview = " ".join(sop_summary.split())[:200] if sop_summary else ""
            trace.append(
                TraceStep(
                    agent="SOPAdvisorAgent",
                    tool=None,
                    summary=(
                        f"Completed using {len(sop_hits)} pre-fetched policy "
                        f"hits. Output: {preview}{'…' if len(sop_summary) > 200 else ''}"
                    ),
                    timestamp=datetime.now(timezone.utc),
                )
            )
        if on_trace_update:
            on_trace_update(list(trace))

        trace.append(
            TraceStep(
                agent="OrchestratorAgent",
                tool=None,
                summary="Synthesizing final report from specialist outputs.",
                timestamp=datetime.now(timezone.utc),
            )
        )
        if on_trace_update:
            on_trace_update(list(trace))

        report_text = await self._synthesize_report(
            zone_id=zone_id,
            hours=hours,
            investigator_summary=incident_summary,
            sop_summary=sop_summary,
            from_date=from_date,
            to_date=to_date,
            sop_hits=sop_hits,
        )
        return report_text, trace

    async def _synthesize_report(
        self,
        *,
        zone_id: str,
        hours: int,
        investigator_summary: str,
        sop_summary: str,
        from_date: Optional[str],
        to_date: Optional[str],
        sop_hits: list[dict[str, str]],
    ) -> str:
        """Build and submit final synthesis prompt to LLM."""
        window_label = (
            f"{from_date} to {to_date}"
            if from_date and to_date
            else f"last {hours} hours"
        )
        clean_investigator_summary = self._normalize_investigator_output(
            self._sanitize_agent_summary(investigator_summary)
        )
        clean_sop_summary = self._sanitize_agent_summary(sop_summary)
        incident_facts = self._parse_incident_facts(clean_investigator_summary)
        timeline_events = self._parse_timeline_events(clean_investigator_summary)
        trend_evidence = self._extract_named_section(
            clean_investigator_summary, "TREND_EVIDENCE"
        )
        data_quality = self._extract_named_section(
            clean_investigator_summary, "DATA_QUALITY"
        )
        sop_actions, compliance_notes = self._parse_sop_recommendations(
            clean_sop_summary
        )
        if not sop_actions:
            sop_actions = self._build_actions_from_hits(sop_hits)
            compliance_notes = self._build_compliance_notes_from_hits(sop_hits)
        severity = (
            incident_facts.get("severity_observed")
            or self._detect_severity(
                clean_investigator_summary, clean_sop_summary
            )
            or "MEDIUM"
        )
        if incident_facts:
            inv_blurb = self._build_facts_blurb(incident_facts)
        elif clean_investigator_summary:
            inv_blurb = self._clean_text(
                clean_investigator_summary[:280]
            )
        else:
            inv_blurb = (
                f"CRITICAL crowd density alerts recorded for zone {zone_id}."
            )
        first_action = (
            sop_actions[0].split(" [")[0]
            if sop_actions
            else "Recommended actions were matched from policy documents."
        )
        executive_summary = self._build_executive_summary(
            zone_id=zone_id,
            window_label=window_label,
            severity=severity,
            investigator_summary=inv_blurb,
            sop_summary=first_action,
        )
        timeline = self._build_timeline_from_events(
            timeline_events=timeline_events,
            fallback_summary=self._build_facts_blurb(incident_facts),
        )
        risk_rows = self._build_risk_rows(
            severity=severity,
            investigator_summary=self._build_risk_rationale(
                incident_facts=incident_facts,
                trend_evidence=trend_evidence,
                data_quality=data_quality,
            ),
            sop_summary=compliance_notes,
        )
        actions = self._build_actions(sop_actions)

        return "\n".join(
            [
                "## INCIDENT REPORT",
                "",
                "**Classification:** OFFICIAL – RESTRICTED",
                "**Prepared by:** Smart City AI Safety System",
                "**Report Version:** 1.0",
                "",
                "## 1. Executive Summary",
                executive_summary,
                "",
                "## 2. Incident Timeline",
                timeline,
                "",
                "## 3. Risk Assessment",
                "",
                "| Risk Factor | Level | Rationale |",
                "|---|---|---|",
                *risk_rows,
                "",
                "## 4. Recommended Actions",
                actions,
                "",
                "## 5. Compliance Notes",
                compliance_notes or "No compliance notes were provided.",
                "",
                "*End of Report*",
            ]
        ).strip()

    @staticmethod
    def _clean_text(value: str) -> str:
        """Collapse repeated whitespace for report-friendly prose."""
        return " ".join((value or "").split()).strip()

    @classmethod
    def _compact_agent_context(cls, value: str, max_chars: int) -> str:
        """Trim long evidence blocks before handing them back to the LLM."""
        clean = (value or "").strip()
        if len(clean) <= max_chars:
            return clean
        head_chars = max(1, int(max_chars * 0.72))
        tail_chars = max(1, max_chars - head_chars - 120)
        return (
            clean[:head_chars].rstrip()
            + "\n\n[... middle evidence compacted to fit LLM context ...]\n\n"
            + clean[-tail_chars:].lstrip()
        )

    @classmethod
    def _summarize_tool_json(cls, raw_result: str) -> str:
        """Return a compact trace summary for JSON tool output."""
        try:
            payload = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError, ValueError):
            return cls._clean_text(raw_result)[:220]
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        if isinstance(rows, list):
            note = payload.get("note") if isinstance(payload, dict) else None
            suffix = f"; {note}" if note else ""
            return f"Retrieved {len(rows)} row(s){suffix}."
        return cls._clean_text(raw_result)[:220]

    async def _run_investigator_tools_direct(
        self,
        *,
        zone_id: str,
        hours: int,
        db_pool: Optional[Any],
        from_date: Optional[str],
        to_date: Optional[str],
    ) -> tuple[str, list[TraceStep]]:
        """Run investigator data tools directly when GAIA returns no calls."""
        if db_pool is None:
            return "", []
        args: dict[str, Any] = {"zone_id": zone_id, "hours": hours}
        if from_date:
            args["from_ts"] = from_date
        if to_date:
            args["to_ts"] = to_date

        trace: list[TraceStep] = []
        for tool_name in (
            "get_zone_alerts",
            "get_density_trends",
            "get_historical_patterns",
        ):
            raw_result = await execute_tool(
                tool_name=tool_name,
                args=args,
                db_pool=db_pool,
                vector_store=None,
                embedder=None,
            )
            trace.append(
                TraceStep(
                    agent="InvestigatorAgent",
                    tool=tool_name,
                    summary=self._summarize_tool_json(raw_result),
                    timestamp=datetime.now(timezone.utc),
                    raw_result=raw_result,
                )
            )

        summary = await self._fallback_investigator_summary(
            zone_id=zone_id,
            hours=hours,
            db_pool=db_pool,
            from_date=from_date,
            to_date=to_date,
        )
        return summary, trace

    @classmethod
    def _sanitize_agent_summary(cls, text: str) -> str:
        """Strip GAIA wrapper text and keep the final answer body."""
        raw = (text or "").strip()
        if not raw:
            return ""

        # Prefer explicit `"answer": "..."` payloads when present.
        answer_match = re.search(
            r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', raw, re.DOTALL
        )
        if answer_match:
            escaped = answer_match.group(1)
            try:
                # Use json.loads to safely unescape the string value.
                decoded = json.loads(f'"{escaped}"')
                if decoded.strip():
                    raw = decoded.strip()
            except (
                AttributeError,
                json.JSONDecodeError,
                ValueError,
            ):
                pass

        message_match = re.search(
            r'"answer"\s*:\s*\{[^{}]*"message"\s*:\s*"((?:\\.|[^"\\])*)"',
            raw,
            re.DOTALL,
        )
        if message_match:
            escaped = message_match.group(1)
            try:
                decoded = json.loads(f'"{escaped}"')
                if decoded.strip():
                    raw = decoded.strip()
            except (
                AttributeError,
                json.JSONDecodeError,
                ValueError,
            ):
                pass

        # Remove common GAIA runtime wrappers/noise.
        noise_patterns = [
            r"⚠️\s*Reached maximum steps limit.*$",
            r"Completed \d+ steps using these tools:.*$",
            r"To continue or complete this task:.*$",
            r'^\{"thought":.*"goal":.*\}\s*$',
        ]
        for pattern in noise_patterns:
            raw = re.sub(pattern, "", raw, flags=re.IGNORECASE | re.DOTALL).strip()

        if raw.startswith("{") and raw.endswith("}"):
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    answer = payload.get("answer")
                    if isinstance(answer, str):
                        return answer.strip()
                    if isinstance(answer, dict):
                        message = answer.get("message")
                        if isinstance(message, str) and message.strip():
                            return message.strip()
            except (
                AttributeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                # Not valid JSON but wrapped in braces — strip them.
                raw = raw[1:-1].strip()

        # Strip stray leading/trailing quote delimiters.
        if raw.startswith('"') and not raw.startswith('"{'):
            raw = raw.lstrip('"').rstrip('"').strip()

        return raw.strip()

    @classmethod
    def _split_sentences(cls, value: str) -> list[str]:
        """Split a block of prose into simple sentence-like chunks."""
        cleaned = cls._clean_text(value)
        if not cleaned:
            return []
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _detect_severity(*values: str) -> str:
        """Return the highest severity label found in the summaries."""
        joined = " ".join(values).upper()
        for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            if level in joined:
                return level
        return "MEDIUM"

    @classmethod
    def _build_executive_summary(
        cls,
        *,
        zone_id: str,
        window_label: str,
        severity: str,
        investigator_summary: str,
        sop_summary: str,
    ) -> str:
        """Create a compact executive summary from specialist outputs."""
        investigator_sentences = cls._split_sentences(investigator_summary)
        sop_sentences = cls._split_sentences(sop_summary)
        first_investigator = (
            investigator_sentences[0]
            if investigator_sentences
            else "No investigator summary was available."
        )
        first_sop = (
            sop_sentences[0]
            if sop_sentences
            else "No SOP guidance was returned."
        )
        return (
            f"This report covers zone {zone_id} for {window_label} and records "
            f"an assessed severity of {severity}. {first_investigator} "
            f"{first_sop}"
        ).strip()

    @classmethod
    def _normalize_investigator_output(cls, text: str) -> str:
        """Convert JSON-format investigator output to markdown section text.

        Handles:
        - Full JSON: {"INCIDENT_FACTS": {...}, ...}
        - Partial JSON (missing outer braces): "INCIDENT_FACTS": {...}, ...
        - Markdown section format (passthrough)
        - Mixed/truncated formats: regex field extraction
        """
        stripped = text.strip()

        # Strip stray leading/trailing quote chars from LLM output.
        if stripped.startswith('"') and not stripped.startswith('"{'):
            stripped = stripped.lstrip('"').rstrip('"').strip()

        # Fast path: already in clean section format.
        if re.search(r"(?:^|\n)INCIDENT_FACTS\s*\n", stripped):
            return stripped

        # Try parsing as full JSON.
        payload: dict[str, Any] = {}
        for attempt in (stripped, f"{{{stripped}}}"):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    payload = obj
                    break
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if not payload and '"INCIDENT_FACTS"' in stripped:
            # Last resort: extract individual fields via regex.
            return cls._regex_extract_incident_text(stripped)

        if not payload:
            return stripped

        lines: list[str] = []
        facts = payload.get("INCIDENT_FACTS", {})
        if isinstance(facts, dict) and facts:
            lines.append("INCIDENT_FACTS")
            for k, v in facts.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        timeline = payload.get("TIMELINE_EVIDENCE", [])
        if timeline:
            lines.append("TIMELINE_EVIDENCE")
            items = (
                timeline if isinstance(timeline, list) else [timeline]
            )
            for item in items:
                if isinstance(item, dict):
                    # {"HH:MM UTC": "event desc"} format
                    for ts, desc in item.items():
                        entry = f"- {ts} — {desc}"
                        lines.append(entry)
                else:
                    entry = str(item).strip()
                    lines.append(
                        entry
                        if entry.startswith("- ")
                        else f"- {entry}"
                    )
            lines.append("")

        trend = payload.get("TREND_EVIDENCE", "")
        if trend:
            lines.append("TREND_EVIDENCE")
            items = trend if isinstance(trend, list) else [trend]
            for item in items:
                entry = str(item).strip()
                lines.append(
                    entry if entry.startswith("- ") else f"- {entry}"
                )
            lines.append("")

        data_quality = payload.get("DATA_QUALITY", {})
        if data_quality:
            lines.append("DATA_QUALITY")
            if isinstance(data_quality, dict):
                for k, v in data_quality.items():
                    lines.append(f"- {k}: {v}")
            else:
                lines.append(f"- {data_quality}")
            lines.append("")

        return "\n".join(lines).strip() if lines else stripped

    @classmethod
    def _regex_extract_incident_text(cls, text: str) -> str:
        """Extract incident fields from unstructured/partial JSON via regex."""
        fields = [
            "zone_id",
            "time_window",
            "total_alerts",
            "severity_observed",
            "min_person_count",
            "max_person_count",
            "latest_alert_utc",
        ]
        facts: dict[str, str] = {}
        for field in fields:
            m = re.search(
                rf'"{re.escape(field)}"\s*:\s*"?([^",\n\}}]+)',
                text,
            )
            if m:
                facts[field] = m.group(1).strip().rstrip('"')

        lines = ["INCIDENT_FACTS"]
        for k, v in facts.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

        # Extract UTC timestamps from timeline
        ts_matches = re.findall(
            r'"(\d{1,2}:\d{2}\s*UTC)"\s*:\s*"([^"]+)"',
            text,
        )
        if ts_matches:
            lines.append("TIMELINE_EVIDENCE")
            for ts, desc in ts_matches[:8]:
                lines.append(f"- {ts} — {desc}")
            lines.append("")

        lines.append("DATA_QUALITY")
        lines.append("- missing_fields: NONE")
        lines.append("- assumptions: NONE")
        return "\n".join(lines)

    @classmethod
    def _extract_named_section(cls, text: str, heading: str) -> str:
        """Extract text block under a heading until the next heading.

        Handles common LLM formatting variations: bold markers, optional
        colons, trailing whitespace, and headings at the start of the string.
        """
        # Flexible: optional ** wrappers, optional colon, trailing spaces.
        heading_pat = rf"\*{{0,2}}{heading}\*{{0,2}}:?[ \t]*"
        next_heading = rf"\*{{0,2}}[A-Z_]{{3,}}\*{{0,2}}:?[ \t]*"
        pattern = (
            rf"(?:^|\n){heading_pat}\n"
            rf"(?P<body>.*?)(?=\n{next_heading}\n|\Z)"
        )
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            return ""
        return cls._clean_text(match.group("body"))

    @classmethod
    def _parse_incident_facts(cls, text: str) -> dict[str, str]:
        """Parse INCIDENT_FACTS key/value bullets."""
        facts_block = cls._extract_named_section(text, "INCIDENT_FACTS")
        facts: dict[str, str] = {}
        if not facts_block:
            return facts
        for line in facts_block.split("- "):
            cleaned = line.strip()
            if not cleaned or ":" not in cleaned:
                continue
            key, value = cleaned.split(":", 1)
            facts[key.strip()] = value.strip()
        return facts

    @classmethod
    def _parse_timeline_events(
        cls, text: str
    ) -> list[tuple[str, str]]:
        """Parse timeline bullet points as (time, description)."""
        events_block = cls._extract_named_section(
            text, "TIMELINE_EVIDENCE"
        )
        events: list[tuple[str, str]] = []
        if not events_block:
            return events
        for chunk in events_block.split("- "):
            item = chunk.strip()
            if not item:
                continue
            if "—" in item:
                ts, desc = item.split("—", 1)
                events.append((ts.strip(), desc.strip()))
                continue
            match = re.match(r"(\d{1,2}:\d{2}\s*UTC)\s*-\s*(.+)", item)
            if match:
                events.append((match.group(1).strip(), match.group(2).strip()))
        return events

    @classmethod
    def _build_timeline_from_events(
        cls,
        *,
        timeline_events: list[tuple[str, str]],
        fallback_summary: str,
    ) -> str:
        """Render timeline bullets from parsed events."""
        if timeline_events:
            return "\n".join(
                f"- {ts} — {cls._clean_text(desc)}"
                for ts, desc in timeline_events[:8]
            )
        if fallback_summary:
            return f"- Summary — {cls._clean_text(fallback_summary)}"
        return "- Summary — No timestamped investigator events were available."

    @classmethod
    def _build_facts_blurb(cls, facts: dict[str, str]) -> str:
        """Build a compact prose sentence from parsed incident facts."""
        if not facts:
            return "No structured incident data were available."
        total_alerts = facts.get("total_alerts", "unknown")
        min_count = facts.get("min_person_count", "unknown")
        max_count = facts.get("max_person_count", "unknown")
        raw_latest = facts.get("latest_alert_utc", "unknown")
        severity = facts.get("severity_observed", "elevated")

        # No data at all for the requested window.
        if str(severity).upper() == "NO_DATA":
            return (
                "No crowd data was recorded for this zone during the "
                "requested time window. The date range may fall outside "
                "the available data retention period, or no sensors were "
                "active for this zone."
            )

        # Avoid double "UTC" when the timestamp already contains it.
        latest = (
            raw_latest.replace(" UTC", "").strip() + " UTC"
            if "UTC" in raw_latest
            else raw_latest
        )
        count_clause = (
            f"between {min_count} and {max_count} persons"
            if min_count != max_count
            else f"{min_count} persons"
        )

        if str(severity).upper() == "SAFE" and str(total_alerts) == "0":
            return (
                f"No threshold violations were recorded during the reporting "
                f"window. Crowd counts of {count_clause} were observed, "
                f"remaining within SAFE limits throughout. "
                f"The most recent observation was at {latest}."
            )

        return (
            f"A total of {total_alerts} {severity.upper()} alert(s) were "
            f"recorded with crowd counts of {count_clause} "
            f"detected during the reporting window. "
            f"The most recent alert occurred at {latest}."
        )

    @classmethod
    def _build_risk_rationale(
        cls,
        *,
        incident_facts: dict[str, str],
        trend_evidence: str,
        data_quality: str,
    ) -> str:
        """Assemble a concise prose risk rationale from incident facts."""
        return cls._build_facts_blurb(incident_facts)

    # Stable, corpus-topical keyword query used when the incident-summary
    # embedding drifts too far from any seeded SOP chunk (e.g. wide date
    # ranges where the investigator narrative is long and discursive).
    # Mirrors the topics of the seeded TX-SCPS-{POL,SOP,TMPL} documents.
    _SOP_FALLBACK_QUERY = (
        "crowd density alert escalation response severity threshold "
        "incident after action report public safety standard operating "
        "procedure data governance privacy notification dispatch"
    )

    @classmethod
    def _retrieve_sop_hits(
        cls,
        *,
        zone_id: str,
        incident_summary: str,
        vector_store: Optional[Any],
        embedder: Optional[Any],
    ) -> list[dict[str, str]]:
        """Retrieve the most relevant SOP/policy chunks for this incident.

        Tries the incident summary as the semantic query first so retrieval
        is driven by what is relevant to the specific event. If that yields
        nothing — which happens when the investigator narrative drifts away
        from the seeded SOP topics (long lookback windows, unusual phrasing,
        etc.) — falls back to a stable corpus-topical keyword query, then
        finally to a generic per-zone query. This guarantees any date range
        produces a usable report as long as the vector store has any policy
        documents at all.
        """
        if vector_store is None or embedder is None:
            return []

        queries: list[str] = []
        primary = cls._compact_agent_context(incident_summary.strip(), 1200)
        if primary:
            queries.append(primary)
        queries.append(cls._SOP_FALLBACK_QUERY)
        queries.append(
            f"crowd density alert escalation response zone {zone_id}"
        )

        for idx, query in enumerate(queries):
            hits = cls._search_sop(query, vector_store, embedder)
            if hits:
                if idx > 0:
                    logger.info(
                        "SOP retrieval used fallback query #%d (primary "
                        "summary embedding returned 0 hits for zone '%s').",
                        idx,
                        zone_id,
                    )
                return hits
        return []

    @classmethod
    def _search_sop(
        cls,
        query: str,
        vector_store: Any,
        embedder: Any,
    ) -> list[dict[str, str]]:
        """Run a single embed+search pass and return up to 6 unique-source hits."""
        try:
            q_emb = embedder.embed_one(query)
            raw_hits = vector_store.search(q_emb, top_k=20)
        except (
            KeyError,
            MilvusException,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning("SOP retrieval failed: %s", exc)
            return []

        best_by_source: dict[str, dict] = {}
        for hit in raw_hits:
            text = cls._clean_text(str(hit.get("text", "")))
            if not text:
                continue
            source = "UNKNOWN_SOURCE"
            try:
                source_path = str(
                    json.loads(str(hit.get("metadata", "{}"))).get("source", "")
                )
                source = source_path.rsplit("/", maxsplit=1)[-1] or source
            except (
                AttributeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                pass
            score = float(hit.get("score") or 0)
            existing = best_by_source.get(source)
            if existing is None or score > float(existing.get("_score", 0)):
                best_by_source[source] = {
                    "source": source,
                    "text": text,
                    "_score": score,
                }

        ranked = sorted(
            best_by_source.values(),
            key=lambda x: x["_score"],
            reverse=True,
        )[:6]
        return [{"source": h["source"], "text": h["text"]} for h in ranked]

    @classmethod
    def _format_sop_hits(cls, hits: list[dict[str, str]]) -> str:
        """Format retrieved SOP chunks for LLM handoff context."""
        lines: list[str] = []
        for idx, hit in enumerate(hits, start=1):
            snippet = hit["text"][:280]
            lines.append(f"{idx}. [{hit['source']}] {snippet}")
        return "\n".join(lines) if lines else "NONE"

    @classmethod
    def _build_timeline(cls, investigator_summary: str) -> str:
        """Extract UTC time markers from the investigator summary."""
        timeline_lines: list[str] = []
        for sentence in cls._split_sentences(investigator_summary):
            match = re.search(r"\b(\d{1,2}:\d{2}\s*UTC)\b", sentence)
            if not match:
                continue
            description = sentence.replace(match.group(1), "").strip(" -:.")
            timeline_lines.append(f"- {match.group(1)} — {description}")

        if timeline_lines:
            return "\n".join(timeline_lines)

        fallback = cls._clean_text(investigator_summary)
        if fallback:
            return f"- Summary — {fallback}"
        return "- Summary — No timestamped investigator events were available."

    @classmethod
    def _build_trend_analysis(
        cls,
        *,
        investigator_summary: str,
        zone_id: str,
        window_label: str,
    ) -> str:
        """Render a narrative trend paragraph from the investigator summary."""
        sentences = cls._split_sentences(investigator_summary)
        if sentences:
            return " ".join(sentences[:2])
        return (
            f"No detailed trend narrative was available for zone {zone_id} "
            f"during {window_label}."
        )

    @classmethod
    def _build_risk_rows(
        cls,
        *,
        severity: str,
        investigator_summary: str,
        sop_summary: str,
    ) -> list[str]:
        """Build the fixed risk-assessment table rows."""
        public_safety = severity
        escalation = "HIGH"
        if "dispatch" not in sop_summary.lower() and severity != "CRITICAL":
            escalation = "MEDIUM"

        return [
            (
                "| Crowd Density | "
                f"{severity} | {cls._clean_text(investigator_summary)} |"
            ),
            (
                "| Public Safety Risk | "
                f"{public_safety} | "
                "Applicable operational and legal obligations are outlined in "
                "the Compliance Notes section. Immediate supervisor and field "
                "unit notification is required for CRITICAL events. |"
            ),
            (
                "| Escalation Potential | "
                f"{escalation} | "
                "Operational intervention is recommended; refer to Recommended "
                "Actions for prioritised response steps. |"
            ),
        ]

    @classmethod
    def _parse_sop_recommendations(
        cls, sop_summary: str
    ) -> tuple[list[str], str]:
        """Parse SOP recommendations and compliance notes from summary."""
        if "NO_SOP_FOUND" in sop_summary:
            return [], "SOP agent returned NO_SOP_FOUND."

        actions_block = cls._extract_named_section(
            sop_summary, "RECOMMENDED_ACTIONS"
        )
        compliance_block = cls._extract_named_section(
            sop_summary, "COMPLIANCE_NOTES"
        )
        actions: list[str] = []
        if actions_block:
            for line in actions_block.splitlines():
                item = line.strip()
                match = re.match(r"^\d+\.\s+(.+)$", item)
                if match:
                    actions.append(cls._clean_text(match.group(1)))

        if not actions:
            # Fallback: accept bullets only if they include citation markers.
            for line in sop_summary.splitlines():
                item = line.strip()
                if item.startswith(("-", "*")):
                    cleaned = cls._clean_text(item.strip("-* "))
                    if "[" in cleaned and "]" in cleaned:
                        actions.append(cleaned)

        compliance_notes = (
            cls._clean_compliance_block(compliance_block)
            if compliance_block
            else "Compliance notes were not provided by SOP retrieval."
        )
        return actions[:5], compliance_notes

    @classmethod
    def _clean_compliance_block(cls, block: str) -> str:
        """Reformat a key: value compliance block into prose bullet lines.

        The SOP agent may return lines like::

            reporting_requirements: Notify supervisors within SLA.
            escalation_thresholds: Activate EOC for CRITICAL events.

        This strips the ``key:`` prefix so each bullet reads as a plain
        sentence.
        """
        lines: list[str] = []
        for raw in block.splitlines():
            stripped = raw.strip("-* \t")
            # Remove "key:" prefix (letters/underscores followed by colon)
            cleaned = re.sub(r"^[a-zA-Z_]+:\s*", "", stripped).strip()
            if cleaned and cleaned.upper() not in ("NONE", "N/A"):
                lines.append(f"- {cleaned}")
        return "\n".join(lines) if lines else block

    @staticmethod
    def _is_table_row_sentence(sentence: str) -> bool:
        """Return True if sentence looks like a threshold table row."""
        if re.match(
            r"^(safe|high|medium|low|critical|emergency)\s+crowd",
            sentence,
            re.IGNORECASE,
        ):
            return True
        if "%" in sentence and "threshold" in sentence.lower():
            return True
        return False

    @staticmethod
    def _extract_threshold_table_action(text: str) -> str:
        """Extract CRITICAL-row action from flattened policy table text."""
        if not text:
            return ""
        pattern = (
            r"CRITICAL\s+Crowd count.*?threshold\s+"
            r"(?P<action>.*?)\s+\d+\s+minutes?\s+EMERGENCY"
        )
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        action = " ".join(match.group("action").split()).strip(" -;,.")
        if ";" in action:
            # Keep the primary imperative clause for concise recommendations.
            action = action.split(";", 1)[0].strip(" -;,.")
        if not action:
            return ""
        return action[0].upper() + action[1:]

    @classmethod
    def _build_actions_from_hits(
        cls, hits: list[dict[str, str]]
    ) -> list[str]:
        """Derive policy-grounded actions from retrieved SOP chunks.

        - One action per unique source document (deduplication).
        - Filters truncated sentences starting with a lowercase character.
        - Filters threshold table rows.
        """
        actions: list[str] = []
        seen_sources: set[str] = set()
        for hit in hits:
            source = hit.get("source", "UNKNOWN_SOURCE")
            if source in seen_sources:
                continue
            text = hit.get("text", "")
            table_action = cls._extract_threshold_table_action(text)
            if table_action:
                actions.append(f"{cls._clean_text(table_action)} [{source}]")
                seen_sources.add(source)
                if len(actions) >= 4:
                    break
                continue
            for sentence in cls._split_sentences(text):
                if sentence and sentence[0].islower():
                    continue
                if cls._is_table_row_sentence(sentence):
                    continue
                lowered = sentence.lower()
                if not any(
                    keyword in lowered
                    for keyword in (
                        "must",
                        "should",
                        "dispatch",
                        "escalat",
                        "notify",
                        "evac",
                        "cordon",
                        "monitor",
                        "notification",
                        "shall",
                        "requir",
                        "activate",
                    )
                ):
                    continue
                actions.append(f"{cls._clean_text(sentence)} [{source}]")
                seen_sources.add(source)
                break
            if len(actions) >= 4:
                break
        if not actions:
            for hit in hits[:3]:
                source = hit.get("source", "UNKNOWN_SOURCE")
                if source in seen_sources:
                    continue
                first_sentence = cls._split_sentences(hit.get("text", ""))
                valid = [
                    s for s in first_sentence
                    if (
                        s
                        and not s[0].islower()
                        and not cls._is_table_row_sentence(s)
                    )
                ]
                if valid:
                    actions.append(
                        f"{cls._clean_text(valid[0])} [{source}]"
                    )
                    seen_sources.add(source)
            if not actions:
                actions = [
                    (
                        "Review cited policy documents and dispatch operations "
                        "response per control requirements. [UNKNOWN_SOURCE]"
                    )
                ]
        return actions

    @classmethod
    def _build_compliance_notes_from_hits(
        cls, hits: list[dict[str, str]]
    ) -> str:
        """Extract specific compliance notes from retrieved SOP chunks.

        Looks for sentences pertaining to:
        - Reporting requirements (notify, log, disclose, audit)
        - Escalation thresholds (threshold exceeded, person count levels)
        - Privacy / civil liberties obligations
        - Data retention obligations
        """
        reporting: list[str] = []
        escalation: list[str] = []
        privacy: list[str] = []
        retention: list[str] = []

        reporting_kw = {
            "notify", "notif", "report", "log", "audit", "disclos",
            "document", "within 60", "within 30", "publish",
        }
        escalation_kw = {
            "escalat", "threshold", "exceed", "critical", "supervisor",
            "field unit", "emergency operations", "activation",
        }
        privacy_kw = {
            "privacy", "civil liberties", "civil rights", "data subject",
            "consent", "purpose limitation", "bias", "disparity",
        }
        retention_kw = {
            "retain", "retention", "days", "purge", "delete", "store",
        }

        seen_sources: set[str] = set()
        for hit in hits:
            source = hit.get("source", "UNKNOWN_SOURCE")
            short_src = source.replace(
                "TX-SCPS-POL-", "POL-"
            ).split(".pdf")[0]
            text = hit.get("text", "")
            table_action = cls._extract_threshold_table_action(text)
            if table_action and len(escalation) < 2:
                escalation.append(
                    f"For CRITICAL threshold events: {table_action} "
                    f"[{short_src}]"
                )
                if (
                    len(reporting) < 2
                    and "notification" in table_action.lower()
                ):
                    reporting.append(
                        f"Notify designated supervisors/field units within "
                        f"threshold SLA. [{short_src}]"
                    )
            for sentence in cls._split_sentences(text):
                if sentence and sentence[0].islower():
                    continue
                if cls._is_table_row_sentence(sentence):
                    continue
                low = sentence.lower()
                clean = cls._clean_text(sentence)
                if (
                    len(reporting) < 2
                    and any(k in low for k in reporting_kw)
                    and source not in {
                        s.split(" [")[1].rstrip("]")
                        for s in reporting
                        if " [" in s
                    }
                ):
                    reporting.append(f"{clean} [{short_src}]")
                elif (
                    len(escalation) < 2
                    and any(k in low for k in escalation_kw)
                    and source not in {
                        s.split(" [")[1].rstrip("]")
                        for s in escalation
                        if " [" in s
                    }
                ):
                    escalation.append(f"{clean} [{short_src}]")
                elif (
                    len(privacy) < 1
                    and any(k in low for k in privacy_kw)
                ):
                    privacy.append(f"{clean} [{short_src}]")
                elif (
                    len(retention) < 1
                    and any(k in low for k in retention_kw)
                ):
                    retention.append(f"{clean} [{short_src}]")

        sources = sorted(
            {hit.get("source", "UNKNOWN_SOURCE") for hit in hits}
        )

        lines: list[str] = []
        if reporting:
            lines.append(f"- {' '.join(reporting)}")
        else:
            lines.append(
                "- Designated supervisors and field units must be notified "
                "in accordance with threshold SLA requirements specified in "
                "cited policy documents."
            )
        if escalation:
            lines.append(f"- {' '.join(escalation)}")
        else:
            lines.append(
                "- Escalation thresholds must be observed as defined in the "
                "cited Standard Operating Procedures."
            )
        if privacy:
            lines.append(f"- {' '.join(privacy)}")
        if retention:
            lines.append(f"- {' '.join(retention)}")
        if sources:
            src_list = ", ".join(sources)
            lines.append(
                f"- All applicable obligations are sourced from: {src_list}."
            )
        return "\n".join(lines)

    @classmethod
    def _build_actions(cls, actions: list[str]) -> str:
        """Render a numbered recommendation list."""
        return "\n".join(
            f"{index}. {cls._clean_text(action)}"
            for index, action in enumerate(actions, start=1)
        )

    @classmethod
    def _build_summary_from_trace(
        cls, investigator_trace: list[TraceStep]
    ) -> str:
        """Reconstruct INCIDENT_FACTS text from tool call results in trace.

        Called when the LLM's final answer cannot be parsed (e.g. thinking
        tokens corrupt the JSON wrapper). The tool call data is always
        present in raw_result regardless of the final-answer format.
        """
        alerts_raw = trends_raw = patterns_raw = None
        for step in investigator_trace:
            if not step.raw_result:
                continue
            if step.tool == "get_zone_alerts" and alerts_raw is None:
                alerts_raw = step.raw_result
            elif step.tool == "get_density_trends" and trends_raw is None:
                trends_raw = step.raw_result
            elif step.tool == "get_historical_patterns" and patterns_raw is None:
                patterns_raw = step.raw_result

        if not alerts_raw:
            return ""  # No tool data — caller must do SQL fallback

        try:
            alerts_data = json.loads(alerts_raw)
            rows = alerts_data.get("rows", [])
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
            return ""

        if not rows:
            return ""

        # Compute aggregate values from alert rows.
        person_counts = [
            r.get("person_count", 0) for r in rows if r.get("person_count")
        ]
        severities = [r.get("severity", "") for r in rows]
        timestamps = [r.get("timestamp", "") for r in rows]
        zone_name = rows[0].get("zone_name", rows[0].get("zone_id", "UNKNOWN"))

        total_alerts = len(rows)
        min_count = min(person_counts) if person_counts else 0
        max_count = max(person_counts) if person_counts else 0
        severity_observed = (
            "CRITICAL" if "CRITICAL" in severities
            else severities[0] if severities else "UNKNOWN"
        )

        # Latest timestamp → HH:MM UTC label.
        latest_label = "UNKNOWN"
        if timestamps:
            try:
                ts = sorted(timestamps)[-1]
                dt = datetime.fromisoformat(
                    ts.replace("+00:00", "").replace("Z", "")
                )
                latest_label = dt.strftime("%H:%M UTC")
            except (AttributeError, TypeError, ValueError):
                latest_label = str(timestamps[-1])[:16]

        # Timeline — last 8 rows (already sorted DESC by tool).
        timeline_lines = []
        for r in rows[:8]:
            try:
                ts = r.get("timestamp", "")
                dt = datetime.fromisoformat(
                    ts.replace("+00:00", "").replace("Z", "")
                )
                t_label = dt.strftime("%H:%M UTC")
            except (AttributeError, TypeError, ValueError):
                t_label = "?:?? UTC"
            sev = r.get("severity", "UNKNOWN")
            pc = r.get("person_count", "?")
            timeline_lines.append(
                f"- {t_label} — {sev} alert detected with {pc} "
                f"persons in zone '{zone_name}'"
            )

        # Trend evidence from density trends if available.
        trend_text = "No hourly trend data available."
        if trends_raw:
            try:
                trends_data = json.loads(trends_raw)
                trend_rows = trends_data.get("rows", [])
                parts = []
                for tr in trend_rows:
                    hb = tr.get("hour_bucket", "")
                    try:
                        dt = datetime.fromisoformat(
                            hb.replace("+00:00", "").replace("Z", "")
                        )
                        h_label = dt.strftime("%H:00")
                    except (AttributeError, TypeError, ValueError):
                        h_label = str(hb)[:5]
                    avg = tr.get("avg_count", 0)
                    peak = tr.get("peak_count", 0)
                    parts.append(
                        f"{h_label} bucket: avg {float(avg):.1f}, peak {peak}"
                    )
                if parts:
                    trend_text = "; ".join(parts)
            except (
                AttributeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                pass

        return (
            "INCIDENT_FACTS\n"
            f"- zone_id: {rows[0].get('zone_id', 'UNKNOWN')}\n"
            f"- time_window: last 24 hours\n"
            f"- total_alerts: {total_alerts}\n"
            f"- severity_observed: {severity_observed}\n"
            f"- min_person_count: {min_count}\n"
            f"- max_person_count: {max_count}\n"
            f"- latest_alert_utc: {latest_label}\n\n"
            "TIMELINE_EVIDENCE\n"
            + "\n".join(timeline_lines)
            + "\n\nTREND_EVIDENCE\n"
            f"- {trend_text}\n\n"
            "DATA_QUALITY\n"
            "- missing_fields: NONE\n"
            "- assumptions: built from InvestigatorAgent tool results\n"
        )

    async def _fallback_investigator_summary(
        self,
        *,
        zone_id: str,
        hours: int,
        db_pool: Optional[Any],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> str:
        """Build structured INCIDENT_FACTS from SQL when GAIA fails.

        Uses absolute from_date / to_date timestamps when provided so that
        historical date-range queries target the correct window rather than
        a rolling offset from NOW().  Falls back to crowd_counts observations
        when crowd_alerts has no rows (SAFE zones).
        """
        if db_pool is None:
            return (
                "INCIDENT_FACTS\n"
                f"- zone_id: {zone_id}\n"
                f"- time_window: {hours} hours\n"
                "- total_alerts: UNKNOWN\n"
                "- severity_observed: UNKNOWN\n"
                "- min_person_count: UNKNOWN\n"
                "- max_person_count: UNKNOWN\n"
                "- latest_alert_utc: UNKNOWN\n\n"
                "TIMELINE_EVIDENCE\n"
                "- Database pool unavailable; no timeline data.\n\n"
                "TREND_EVIDENCE\n"
                "- No trend data available.\n\n"
                "DATA_QUALITY\n"
                "- missing_fields: all (database unavailable)\n"
                "- assumptions: NONE\n"
            )

        # Build the time-window filter: prefer absolute timestamps.
        now = datetime.now(timezone.utc)
        if from_date and to_date:
            try:
                ts_start = datetime.fromisoformat(from_date).replace(
                    tzinfo=timezone.utc
                )
                ts_end = datetime.fromisoformat(to_date).replace(
                    tzinfo=timezone.utc
                ) + timedelta(days=1)
                ts_end = min(ts_end, now)
                window_label = f"{from_date} to {to_date}"
                use_absolute = True
            except ValueError:
                use_absolute = False
        else:
            use_absolute = False

        if use_absolute:
            agg_query = (
                "SELECT COUNT(*) AS alert_count, "
                "MIN(person_count) AS min_count, "
                "MAX(person_count) AS max_count, "
                "MAX(severity) AS severity, "
                "MAX(timestamp) AS latest_ts "
                "FROM crowd_alerts "
                "WHERE zone_id=$1 AND timestamp >= $2 AND timestamp < $3"
            )
            rows_query = (
                "SELECT timestamp, zone_name, severity, person_count "
                "FROM crowd_alerts "
                "WHERE zone_id=$1 AND timestamp >= $2 AND timestamp < $3 "
                "ORDER BY timestamp DESC LIMIT 8"
            )
            trend_query = (
                "SELECT date_trunc('hour', timestamp) AS hour_bucket, "
                "AVG(person_count)::float AS avg_count, "
                "MAX(person_count) AS peak_count "
                "FROM crowd_alerts "
                "WHERE zone_id=$1 AND timestamp >= $2 AND timestamp < $3 "
                "GROUP BY 1 ORDER BY 1"
            )
            counts_agg_query = (
                "SELECT COUNT(*) AS obs_count, "
                "MIN(person_count) AS min_count, "
                "MAX(person_count) AS max_count, "
                "MAX(timestamp) AS latest_ts "
                "FROM crowd_counts "
                "WHERE zone_id=$1 AND timestamp >= $2 AND timestamp < $3"
            )
            counts_rows_query = (
                "SELECT timestamp, zone_name, person_count, severity "
                "FROM crowd_counts "
                "WHERE zone_id=$1 AND timestamp >= $2 AND timestamp < $3 "
                "ORDER BY timestamp DESC LIMIT 8"
            )
            q_args: tuple = (zone_id, ts_start, ts_end)
        else:
            window_label = f"last {hours} hours"
            agg_query = (
                "SELECT COUNT(*) AS alert_count, "
                "MIN(person_count) AS min_count, "
                "MAX(person_count) AS max_count, "
                "MAX(severity) AS severity, "
                "MAX(timestamp) AS latest_ts "
                "FROM crowd_alerts "
                "WHERE zone_id=$1 "
                "  AND timestamp >= NOW() - INTERVAL '1 hour' * $2"
            )
            rows_query = (
                "SELECT timestamp, zone_name, severity, person_count "
                "FROM crowd_alerts "
                "WHERE zone_id=$1 "
                "  AND timestamp >= NOW() - INTERVAL '1 hour' * $2 "
                "ORDER BY timestamp DESC LIMIT 8"
            )
            trend_query = (
                "SELECT date_trunc('hour', timestamp) AS hour_bucket, "
                "AVG(person_count)::float AS avg_count, "
                "MAX(person_count) AS peak_count "
                "FROM crowd_alerts "
                "WHERE zone_id=$1 "
                "  AND timestamp >= NOW() - INTERVAL '1 hour' * $2 "
                "GROUP BY 1 ORDER BY 1"
            )
            counts_agg_query = (
                "SELECT COUNT(*) AS obs_count, "
                "MIN(person_count) AS min_count, "
                "MAX(person_count) AS max_count, "
                "MAX(timestamp) AS latest_ts "
                "FROM crowd_counts "
                "WHERE zone_id=$1 "
                "  AND timestamp >= NOW() - INTERVAL '1 hour' * $2"
            )
            counts_rows_query = (
                "SELECT timestamp, zone_name, person_count, severity "
                "FROM crowd_counts "
                "WHERE zone_id=$1 "
                "  AND timestamp >= NOW() - INTERVAL '1 hour' * $2 "
                "ORDER BY timestamp DESC LIMIT 8"
            )
            q_args = (zone_id, hours)

        try:
            async with db_pool.acquire() as conn:
                agg = await conn.fetchrow(agg_query, *q_args)
                rows = await conn.fetch(rows_query, *q_args)
                trend_rows = await conn.fetch(trend_query, *q_args)
                # Always fetch crowd_counts for SAFE-zone fallback.
                counts_agg = await conn.fetchrow(counts_agg_query, *q_args)
                counts_rows = await conn.fetch(counts_rows_query, *q_args)
        except (
            asyncpg.PostgresError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning("SQL fallback summary failed: %s", exc)
            return (
                "INCIDENT_FACTS\n"
                f"- zone_id: {zone_id}\n"
                f"- time_window: {window_label}\n"
                "- total_alerts: UNKNOWN\n"
                "- severity_observed: UNKNOWN\n"
                "- min_person_count: UNKNOWN\n"
                "- max_person_count: UNKNOWN\n"
                "- latest_alert_utc: UNKNOWN\n\n"
                "TIMELINE_EVIDENCE\n"
                "- SQL query failed; no timeline data.\n\n"
                "TREND_EVIDENCE\n"
                "- No trend data available.\n\n"
                "DATA_QUALITY\n"
                "- missing_fields: all (query error)\n"
                "- assumptions: NONE\n"
            )

        alert_count = int(agg["alert_count"] or 0) if agg else 0
        if alert_count == 0:
            # No threshold violations — use crowd_counts observations.
            obs_count = int(counts_agg["obs_count"] or 0) if counts_agg else 0
            if obs_count == 0:
                return (
                    "INCIDENT_FACTS\n"
                    f"- zone_id: {zone_id}\n"
                    f"- time_window: {window_label}\n"
                    "- total_alerts: 0\n"
                    "- severity_observed: NO_DATA\n"
                    "- min_person_count: UNKNOWN\n"
                    "- max_person_count: UNKNOWN\n"
                    "- latest_alert_utc: UNKNOWN\n\n"
                    "TIMELINE_EVIDENCE\n"
                    f"- No data recorded for zone {zone_id} "
                    f"in window {window_label}.\n\n"
                    "TREND_EVIDENCE\n"
                    "- No data available for this zone and time window.\n\n"
                    "DATA_QUALITY\n"
                    "- missing_fields: all\n"
                    "- assumptions: NONE\n"
                )
            # Observations available — report as SAFE
            latest_obs_ts = counts_agg["latest_ts"]
            latest_obs_label = (
                latest_obs_ts.strftime("%H:%M UTC")
                if latest_obs_ts
                else "UNKNOWN"
            )
            obs_zone = counts_rows[0]["zone_name"] if counts_rows else zone_id
            timeline_lines = [
                f"- {r['timestamp'].strftime('%H:%M UTC')} — "
                f"SAFE observation: {r['person_count']} persons in "
                f"zone '{obs_zone}'"
                for r in counts_rows
                if r["timestamp"]
            ]
            return (
                "INCIDENT_FACTS\n"
                f"- zone_id: {zone_id}\n"
                f"- time_window: {window_label}\n"
                f"- total_alerts: 0\n"
                "- severity_observed: SAFE\n"
                f"- min_person_count: {int(counts_agg['min_count'] or 0)}\n"
                f"- max_person_count: {int(counts_agg['max_count'] or 0)}\n"
                f"- latest_alert_utc: {latest_obs_label}\n\n"
                "TIMELINE_EVIDENCE\n"
                + "\n".join(timeline_lines or [
                    "- No timestamped observations available."
                ])
                + "\n\nTREND_EVIDENCE\n"
                f"- Zone maintained SAFE status throughout {window_label} "
                f"with {obs_count} observations recorded. "
                f"Person counts ranged from "
                f"{int(counts_agg['min_count'] or 0)} to "
                f"{int(counts_agg['max_count'] or 0)}.\n\n"
                "DATA_QUALITY\n"
                "- missing_fields: NONE\n"
                "- assumptions: crowd_counts used (no threshold violations)\n"
            )

        latest_ts = agg["latest_ts"]
        latest_label = (
            latest_ts.strftime("%H:%M UTC") if latest_ts else "UNKNOWN"
        )

        zone_name = rows[0]["zone_name"] if rows else zone_id

        timeline_lines = []
        for r in rows:
            ts_label = (
                r["timestamp"].strftime("%H:%M UTC")
                if r["timestamp"]
                else "UNKNOWN"
            )
            timeline_lines.append(
                f"- {ts_label} — {r['severity']} alert detected with "
                f"{r['person_count']} persons in zone '{zone_name}'"
            )

        trend_parts = []
        displayed_trends = list(trend_rows[:12])
        if len(trend_rows) > 12:
            displayed_trends.extend(trend_rows[-6:])
        seen_hours: set[Any] = set()
        for tr in displayed_trends:
            hour_bucket = tr["hour_bucket"]
            if hour_bucket in seen_hours:
                continue
            seen_hours.add(hour_bucket)
            hour_label = (
                hour_bucket.strftime("%Y-%m-%d %H:00")
                if hour_bucket
                else "??"
            )
            trend_parts.append(
                f"{hour_label} hour bucket: avg {tr['avg_count']:.1f}, "
                f"peak {tr['peak_count']}"
            )
        if len(trend_rows) > len(seen_hours):
            trend_parts.append(
                f"{len(trend_rows) - len(seen_hours)} additional hourly "
                "buckets compacted; aggregate min/max are in INCIDENT_FACTS"
            )
        trend_text = (
            "; ".join(trend_parts)
            if trend_parts
            else "No hourly trend data available."
        )

        return (
            "INCIDENT_FACTS\n"
            f"- zone_id: {zone_id}\n"
            f"- time_window: {window_label}\n"
            f"- total_alerts: {int(agg['alert_count'])}\n"
            f"- severity_observed: {agg['severity'] or 'UNKNOWN'}\n"
            f"- min_person_count: {int(agg['min_count'] or 0)}\n"
            f"- max_person_count: {int(agg['max_count'] or 0)}\n"
            f"- latest_alert_utc: {latest_label}\n\n"
            "TIMELINE_EVIDENCE\n"
            + "\n".join(timeline_lines)
            + "\n\nTREND_EVIDENCE\n"
            f"- {trend_text}\n\n"
            "DATA_QUALITY\n"
            "- missing_fields: NONE\n"
            "- assumptions: NONE\n"
        )
