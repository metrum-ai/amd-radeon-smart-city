# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""LLM-powered incident report generator."""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, List, Optional

import asyncpg
import httpx

from smart_city.llm.agents import OrchestratorAgent, TraceStep
from smart_city.llm.embedder import Embedder
from smart_city.llm.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks from Qwen3/3.5 model output.

    Lemonade's llama.cpp backend may return thinking tokens in the content
    field when the model is in thinking mode.  This strips them so only
    the final answer reaches the report renderer.

    Args:
        text: Raw LLM response string.

    Returns:
        Text with all thinking blocks removed and leading/trailing
        whitespace stripped.
    """
    return _THINK_RE.sub("", text).strip()

_REPORT_SYSTEM_PROMPT = """\
You are a professional public-safety intelligence analyst for a Smart City
operations centre.

Produce a formal INCIDENT REPORT using ONLY the crowd-event data supplied.
Do not invent figures, locations, or timestamps.
Use third-person, formal language throughout.
After your analysis/reasoning, output the final report using this exact
Markdown structure:

---
## INCIDENT REPORT

**Classification:** OFFICIAL – RESTRICTED
**Prepared by:** Smart City AI Safety System
**Report Version:** 1.0

---

## 1. Executive Summary
One concise paragraph: nature of event, zone affected, severity, and period
covered.

## 2. Incident Timeline
Chronological bullet list of key events extracted from the data.
Format each line as: `HH:MM UTC — <description>`

## 3. Risk Assessment

| Risk Factor          | Level                    | Rationale |
|----------------------|--------------------------|-----------|
| Crowd Density        | CRITICAL / HIGH / MEDIUM / LOW | … |
| Public Safety Risk   | CRITICAL / HIGH / MEDIUM / LOW | … |
| Escalation Potential | HIGH / MEDIUM / LOW      | … |

## 4. Recommended Actions
Numbered, prioritised list of concrete operational recommendations for
security personnel and city management.

---
*End of Report*
"""


class ReportGenerator:
    """Generate PDF-ready incident reports using LLM + RAG.

    Pulls recent alerts from TimescaleDB and similar historical events
    from Milvus, then asks the LLM to produce a structured narrative.
    """

    def __init__(
        self,
        vllm_base_url: str,
        model_name: str,
        embedder: Embedder,
        vector_store: VectorStoreClient,
        db_pool: Optional[Any] = None,
        max_tokens: int = 1024,
        api_key: str = "",
    ) -> None:
        """Initialise the report generator.

        Args:
            vllm_base_url: Base URL of the OpenAI-compatible API.
                For OpenRouter use https://openrouter.ai/api/v1.
            model_name: Model identifier (vLLM alias or OpenRouter slug,
                e.g. ``openai/gpt-4o``).
            embedder: Sentence-transformer embedder instance.
            vector_store: Connected Milvus vector-store client.
            db_pool: Optional asyncpg connection pool for SQL context.
            max_tokens: Maximum tokens for the LLM response.
            api_key: Bearer token for hosted providers (e.g. OpenRouter).
                Leave empty for local vLLM deployments that need no auth.
        """
        # Normalise: strip whitespace/trailing slash, then remove a
        # trailing /v1 so we can always append /v1/chat/completions
        # uniformly (handles both http://vllm:8001 and
        # https://openrouter.ai/api/v1).
        _url = vllm_base_url.strip().rstrip("/")
        self._vllm_url = _url[:-3] if _url.endswith("/v1") else _url
        self._model = model_name
        self._api_key = api_key
        self._embedder = embedder
        self._vector_store = vector_store
        self._db_pool = db_pool
        self._max_tokens = max_tokens
        self.agent_trace: List[TraceStep] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_incident_report(
        self,
        zone_id: str,
        hours: int = 24,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        on_trace_update: Optional[Any] = None,
    ) -> str:
        """Generate a structured incident report for a zone.

        SQL event loading and RAG retrieval run concurrently to reduce
        overall latency.

        Args:
            zone_id: Target zone identifier.
            hours: Effective lookback window (computed from date range
                when ``from_date``/``to_date`` are supplied).
            from_date: Optional ISO date string for window start.
            to_date: Optional ISO date string for window end.

        Returns:
            Markdown-formatted report string.
        """
        orchestrator = OrchestratorAgent(
            vllm_url=self._vllm_url,
            model=self._model,
            api_key=self._api_key,
        )
        report_text, trace = await orchestrator.run(
            zone_id=zone_id,
            hours=hours,
            db_pool=self._db_pool,
            vector_store=self._vector_store,
            embedder=self._embedder,
            from_date=from_date,
            to_date=to_date,
            on_trace_update=on_trace_update,
        )
        self.agent_trace = trace
        content = self._extract_final_report(report_text or "")
        if "## INCIDENT REPORT" not in content:
            preview = (report_text or "")[:600].replace("\n", " ")
            raise RuntimeError(
                f"LLM response missing final report section. "
                f"Raw output preview: {preview!r}"
            )
        return content

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _load_sql_events(
        self, zone_id: str, hours: int
    ) -> List[str]:
        """Load recent alerts from TimescaleDB for a zone.

        Args:
            zone_id: Zone identifier to filter on.
            hours: Lookback window.

        Returns:
            List of formatted event strings.
        """
        events: List[str] = []
        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT timestamp, zone_name, zone_type, "
                    "violation_type, person_count, threshold, "
                    "severity, description "
                    "FROM crowd_alerts "
                    "WHERE zone_id=$1 "
                    "  AND timestamp >= NOW() - INTERVAL '1 hour' * $2 "
                    "ORDER BY timestamp DESC LIMIT 50",
                    zone_id,
                    hours,
                )
            for r in rows:
                events.append(
                    f"[{r['timestamp']}] {r['zone_name']}: "
                    f"{r['person_count']} people detected, "
                    f"type={r['violation_type'] or r['zone_type']}, "
                    f"severity={r['severity']}. "
                    f"{r['description'] or ''}"
                )
        except (asyncpg.PostgresError, OSError, ValueError) as exc:
            logger.error(
                "Failed to load SQL events: %s", exc, exc_info=True
            )
        return events

    def _retrieve_rag(self, query: str) -> List[str]:
        """Retrieve RAG context from Milvus (blocking).

        Args:
            query: Search query string.

        Returns:
            List of relevant text chunks.
        """
        qvec = self._embedder.embed_one(query)
        hits = self._vector_store.search(qvec, top_k=5)
        return [h["text"] for h in hits if h.get("text")]

    def _extract_final_report(self, raw_text: str) -> str:
        """Strip model reasoning and keep only final report markdown.

        Handles three thinking-token patterns that leak into content:
        - ``<think>...</think>`` XML blocks (some open-source models)
        - ``Thinking Process:`` plain-text preambles
        - ``## INCIDENT REPORT`` is always used as the canonical start marker.

        Args:
            raw_text: Raw model output text.

        Returns:
            Markdown beginning at ``## INCIDENT REPORT`` when present.
        """
        import re

        text = raw_text.strip()

        # Strip <think>…</think> blocks that some models emit in content.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Strip leading "Thinking Process:" preamble up to the first blank line
        # followed by non-thinking content.
        if text.startswith("Thinking Process:"):
            # Drop everything before the first occurrence of ## INCIDENT REPORT
            pass  # fall through to start_marker search below

        start_marker = "## INCIDENT REPORT"
        start_idx = text.find(start_marker)
        if start_idx >= 0:
            text = text[start_idx:]

        end_marker = "*End of Report*"
        end_idx = text.find(end_marker)
        if end_idx >= 0:
            text = text[: end_idx + len(end_marker)]

        return text.strip()

    async def _call_llm(self, prompt: str) -> str:
        """Call the OpenAI-compatible completions endpoint.

        Works with local vLLM and hosted providers such as OpenRouter.
        An ``Authorization: Bearer`` header is added automatically when
        ``api_key`` is non-empty.

        Args:
            prompt: Full prompt string.

        Returns:
            Generated report text.
        """
        try:
            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": _REPORT_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self._max_tokens,
                "temperature": 0.2,
            }
            # OpenRouter: exclude reasoning tokens from response body
            if self._api_key:
                payload["reasoning"] = {"exclude": True}
            headers: dict = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{self._vllm_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                # Lemonade puts thinking in reasoning_content, final answer in
                # content. When content is empty (thinking-only response), fall
                # back to reasoning_content and strip <think> blocks.
                raw_content = msg.get("content") or ""
                if not raw_content:
                    raw_content = (
                        msg.get("reasoning_content")
                        or msg.get("reasoning")
                        or ""
                    )
                # Strip any residual <think>…</think> blocks regardless of source
                raw_content = _strip_thinking(str(raw_content))
                content = self._extract_final_report(raw_content)
                if "## INCIDENT REPORT" not in content:
                    raise RuntimeError(
                        "LLM response missing final report section."
                    )
                return content
        except (
            httpx.HTTPError,
            json.JSONDecodeError,
            KeyError,
            RuntimeError,
            ValueError,
        ) as exc:
            logger.error("Report LLM call failed: %s", exc, exc_info=True)
            return (
                f"## Report Generation Failed\n\n"
                f"Could not generate report: {exc}"
            )
