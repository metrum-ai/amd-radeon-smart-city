# Created by Metrum AI for AMD

"""Report generation and retrieval endpoints.

Reports are rendered to PDF and stored under ``/app/data/reports/``
(configurable via the ``REPORTS_DIR`` environment variable).
No database persistence is used for report content.

Filename convention::

    {report_id}__{zone_id}__{YYYYMMDD_HHMMSS}.pdf

The double-underscore separators allow zone IDs that contain single
underscores.

Generation is asynchronous: ``POST /reports/generate`` returns a job
immediately (status ``pending``); the LLM + PDF work runs in the
background.  Poll ``GET /reports/{report_id}/status`` until status is
``completed`` or ``failed``, then fetch the PDF via
``GET /reports/{report_id}/download``.

Agent handoffs are visible via ``GET /reports/{report_id}/trace``.
"""

import asyncio
import json
import logging
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import asyncpg
import httpx
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

# Strict UUID v4 shape (8-4-4-4-12 hex chars). Used to validate the
# ``report_id`` path parameter before any filesystem operation so a
# crafted value like "../../etc/passwd" cannot reach Path.glob().
_REPORT_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                           r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _validate_report_id(report_id: str) -> str:
    """Reject any report_id that isn't a canonical UUID string.

    Args:
        report_id: Value supplied via the URL path.

    Returns:
        The validated report_id (unchanged).

    Raises:
        HTTPException: 400 when the value is not a UUID.
    """
    if not isinstance(report_id, str) or not _REPORT_ID_RE.fullmatch(report_id):
        raise HTTPException(status_code=400, detail="Invalid report_id.")
    return report_id

from smart_city.api.models import (
    AgentTraceStep,
    ReportJobResponse,
    ReportRequest,
    ReportResponse,
)

_STREAMS_METADATA_FILE = os.environ.get(
    "STREAMS_METADATA_FILE", "/app/smart_city/config/streams_metadata.yaml"
)
_location_cache: Optional[Dict[str, dict]] = None


def _load_location_index() -> Dict[str, dict]:
    """Build a zone_id → location dict from streams_metadata.yaml (cached).

    Indexes by both the explicit ``zone_id`` field and the ``cam{id}_main``
    pattern so that pipeline-generated zone IDs resolve correctly.
    """
    global _location_cache  # pylint: disable=global-statement
    if _location_cache is not None:
        return _location_cache
    from smart_city.core.stream_allocation import (  # noqa: PLC0415
        apply_city_allocation,
    )

    try:
        with open(_STREAMS_METADATA_FILE, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        streams = raw.get("streams", raw) if isinstance(raw, dict) else raw
        if isinstance(streams, list):
            streams = apply_city_allocation(list(streams))
            idx: Dict[str, dict] = {}
            for s in streams:
                meta = {
                    "location_name": s.get("location_name", ""),
                    "city": s.get("city", ""),
                    "lat": s.get("lat"),
                    "lon": s.get("lon"),
                }
                if s.get("zone_id"):
                    idx[str(s["zone_id"])] = meta
                if s.get("id"):
                    # Also index as cam{id}_main for pipeline zone IDs
                    idx[f"cam{s['id']}_main"] = meta
                    idx[f"cam{s['id']}"] = meta
            _location_cache = idx
        else:
            _location_cache = {}
    except (OSError, yaml.YAMLError, ValueError, KeyError) as exc:
        logger.warning(
            "Could not load streams metadata from %s: %s",
            _STREAMS_METADATA_FILE, exc, exc_info=True,
        )
        _location_cache = {}
    return _location_cache


def _resolve_location(zone_id: str) -> dict:
    """Return location metadata for a zone_id, with safe defaults."""
    idx = _load_location_index()
    # Direct match first; fall back to prefix match (e.g. cam1_main → cam1)
    meta = idx.get(zone_id)
    if meta is None:
        prefix = zone_id.split("_")[0]
        meta = next(
            (v for k, v in idx.items() if k.startswith(prefix)), None
        )
    return meta or {"location_name": zone_id, "city": None, "lat": None, "lon": None}


router = APIRouter(tags=["reports"])

REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "/app/data/reports"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _report_filename(report_id: str, zone_id: str, ts: datetime) -> str:
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    return f"{report_id}__{zone_id}__{ts_str}.pdf"


def _parse_filename(name: str) -> Optional[dict]:
    if not name.endswith(".pdf"):
        return None
    stem = name[:-4]
    parts = stem.split("__", 2)
    if len(parts) != 3:
        return None
    report_id, zone_id, ts_str = parts
    try:
        generated_at = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return {"report_id": report_id, "zone_id": zone_id, "generated_at": generated_at}


def _find_pdf(report_id: str) -> Optional[Path]:
    """Locate the PDF for a previously generated report.

    ``report_id`` is assumed to have already passed :func:`_validate_report_id`,
    but we re-check defensively so this helper is safe to call from any path.
    Resolved paths are also asserted to live under ``REPORTS_DIR`` to defeat
    symlink/traversal escape.
    """
    if not _REPORT_ID_RE.fullmatch(report_id):
        return None
    base = REPORTS_DIR.resolve()
    for pdf in REPORTS_DIR.glob(f"{report_id}__*.pdf"):
        try:
            resolved = pdf.resolve()
        except OSError:
            continue
        if base == resolved.parent or base in resolved.parents:
            return pdf
    return None


def _get_jobs(request: Request) -> Dict[str, dict]:
    """Return the in-memory job registry from app state."""
    return request.app.state.report_jobs  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


_AGENT_DISPATCH_LABELS: Dict[str, str] = {
    "InvestigatorAgent": "Fetch alerts, density trends, and patterns for zone",
    "SOPAdvisorAgent": "Retrieve SOP policy guidance and build recommendations",
    "OrchestratorAgent": "Synthesize specialist outputs into final report",
}


async def _run_report_job(
    app,
    report_id: str,
    zone_id: str,
    hours: int,
    from_date: Optional[str],
    to_date: Optional[str],
    location_name: Optional[str],
) -> None:
    """Generate the report, render PDF, and update job state."""
    jobs: dict = app.state.report_jobs
    jobs[report_id]["status"] = "processing"

    try:
        generator = app.state.report_generator
        if generator is None:
            raise RuntimeError("Report generator not ready.")

        def _flush_trace(partial_steps: list) -> None:
            """Write partial trace steps to job state for live polling."""
            loc_label = location_name or zone_id
            formatted = []
            for step in partial_steps:
                # Only show input_preview for tool-call steps to avoid
                # duplicating the summary text on delegation/dispatch steps.
                if step.tool is not None:
                    label = _AGENT_DISPATCH_LABELS.get(step.agent, "")
                    input_preview: str | None = (
                        f"{label} @ {loc_label}" if label else None
                    )
                else:
                    input_preview = None
                formatted.append({
                    "agent": step.agent,
                    "tool": step.tool,
                    "summary": step.summary,
                    "timestamp": step.timestamp.isoformat(),
                    "input_preview": input_preview,
                })
            jobs[report_id]["agent_trace"] = formatted

        content = await generator.generate_incident_report(
            zone_id=zone_id,
            hours=hours,
            from_date=from_date,
            to_date=to_date,
            on_trace_update=_flush_trace,
        )
        if content.lstrip().startswith("## Report Generation Failed"):
            raise RuntimeError(content.strip())

        generated_at = datetime.now(tz=timezone.utc)

        from smart_city.llm.pdf_renderer import markdown_to_pdf  # noqa: PLC0415

        pdf_bytes = markdown_to_pdf(
            markdown_text=content,
            zone_id=zone_id,
            generated_at=generated_at.strftime("%Y-%m-%d %H:%M UTC"),
        )
        _ensure_reports_dir()
        filename = _report_filename(report_id, zone_id, generated_at)
        (REPORTS_DIR / filename).write_bytes(pdf_bytes)
        logger.info("Report saved: %s (%d bytes)", filename, len(pdf_bytes))

        raw_trace = getattr(generator, "agent_trace", []) or []
        loc_label = location_name or zone_id
        trace_steps = []
        for step in raw_trace:
            # Only show input_preview for tool-call steps; delegation/dispatch
            # steps carry their full context in summary already.
            if step.tool is not None:
                label = _AGENT_DISPATCH_LABELS.get(step.agent, "")
                input_preview = (
                    f"{label} @ {loc_label}" if label else None
                )
            else:
                input_preview = None
            trace_steps.append(
                {
                    "agent": step.agent,
                    "tool": step.tool,
                    "summary": step.summary,
                    "timestamp": step.timestamp,
                    "input_preview": input_preview,
                }
            )

        jobs[report_id].update(
            {
                "status": "completed",
                "generated_at": generated_at,
                "content": content,
                "agent_trace": trace_steps,
            }
        )

    except (
        asyncpg.PostgresError,
        httpx.HTTPError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        logger.error(
            "Report job %s failed: %s", report_id, exc, exc_info=True
        )
        jobs[report_id].update({"status": "failed", "error": str(exc)})
    except Exception as exc:  # pylint: disable=broad-except
        # Final safety net for any unanticipated failure mode in the
        # background task — without this the asyncio task would die
        # silently and the job would be stuck in "processing" forever.
        logger.exception("Report job %s crashed unexpectedly", report_id)
        jobs[report_id].update({"status": "failed", "error": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/reports/generate", response_model=ReportJobResponse, status_code=202)
async def generate_report(
    body: ReportRequest, request: Request
) -> ReportJobResponse:
    """Submit an async report generation job.

    Returns immediately with ``status: pending`` and a stable
    ``report_id``.  The LLM inference and PDF rendering happen in the
    background.  Poll ``GET /reports/{report_id}/status`` to track
    progress.

    Args:
        body: Zone ID and optional look-back hours / date range.
        request: FastAPI request (used to access app state).

    Returns:
        Job metadata including the stable ``report_id``.

    Raises:
        HTTPException: 503 if the report generator is not ready.
        HTTPException: 422 if date format is invalid.
    """
    if getattr(request.app.state, "report_generator", None) is None:
        raise HTTPException(status_code=503, detail="Report generator not ready.")

    # Resolve effective hours from date range when supplied
    effective_hours: int = body.hours
    if body.from_date and body.to_date:
        try:
            from datetime import date  # noqa: PLC0415

            dt_from = date.fromisoformat(body.from_date)
            dt_to = date.fromisoformat(body.to_date)
            today = date.today()
            if dt_from > dt_to:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"from_date ({body.from_date}) must not be "
                        f"after to_date ({body.to_date})."
                    ),
                )
            if dt_from > today:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"from_date ({body.from_date}) is in the future; "
                        "no data will exist for that range."
                    ),
                )
            delta_hours = max(
                1,
                math.ceil((dt_to - dt_from).total_seconds() / 3600),
            )
            effective_hours = min(delta_hours, 24 * 365)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid date format: {exc}"
            ) from exc

    report_id = str(uuid.uuid4())
    submitted_at = datetime.now(tz=timezone.utc)

    # Resolve location metadata from streams_metadata.yaml
    loc = _resolve_location(body.zone_id)
    resolved_name = body.location_name or loc["location_name"] or body.zone_id

    _get_jobs(request)[report_id] = {
        "status": "pending",
        "zone_id": body.zone_id,
        "location_name": resolved_name,
        "city": loc["city"],
        "lat": loc["lat"],
        "lon": loc["lon"],
        "submitted_at": submitted_at,
        "generated_at": None,
        "error": None,
        "content": None,
        "agent_trace": None,
    }

    asyncio.create_task(
        _run_report_job(
            app=request.app,
            report_id=report_id,
            zone_id=body.zone_id,
            hours=effective_hours,
            from_date=body.from_date,
            to_date=body.to_date,
            location_name=resolved_name,
        )
    )

    return ReportJobResponse(
        report_id=report_id,
        status="pending",
        zone_id=body.zone_id,
        location_name=resolved_name,
        city=loc["city"],
        lat=loc["lat"],
        lon=loc["lon"],
        submitted_at=submitted_at,
    )


@router.get("/reports/{report_id}/status", response_model=ReportJobResponse)
async def get_report_status(
    report_id: str, request: Request
) -> ReportJobResponse:
    """Poll the status of an async report generation job.

    Args:
        report_id: UUID returned by ``POST /reports/generate``.
        request: FastAPI request (used to access app state).

    Returns:
        Current job status.  ``status`` is one of:
        ``pending`` | ``processing`` | ``completed`` | ``failed``.

    Raises:
        HTTPException: 400 if report_id is not a valid UUID.
        HTTPException: 404 if the report_id is unknown.
    """
    _validate_report_id(report_id)
    jobs = _get_jobs(request)
    job = jobs.get(report_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Report job not found.")

    return ReportJobResponse(report_id=report_id, **job)


@router.get("/reports/{report_id}/trace", response_model=List[AgentTraceStep])
async def get_report_trace(
    report_id: str, request: Request
) -> List[AgentTraceStep]:
    """Return the agent handoff trace for a completed report.

    Each step shows which agent ran, which tool it called, a summary of
    what was retrieved or computed, and the input that triggered it.
    This lets you see the full InvestigatorAgent → SOPAdvisorAgent →
    OrchestratorAgent handoff chain.

    Args:
        report_id: UUID returned by ``POST /reports/generate``.
        request: FastAPI request (used to access app state).

    Returns:
        Ordered list of agent trace steps.  Empty list while still
        ``pending`` or ``processing``.

    Raises:
        HTTPException: 400 if report_id is not a valid UUID.
        HTTPException: 404 if the report_id is unknown.
    """
    _validate_report_id(report_id)
    jobs = _get_jobs(request)
    job = jobs.get(report_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Report job not found.")
    return [AgentTraceStep(**step) for step in (job.get("agent_trace") or [])]


@router.get("/reports", response_model=List[ReportResponse])
async def list_reports(request: Request) -> List[ReportResponse]:  # pylint: disable=unused-argument
    """Return the 20 most recent completed reports (newest first).

    Content is empty in list responses; use
    ``GET /reports/{report_id}/download`` for the full PDF.
    """
    _ensure_reports_dir()
    entries = []
    for pdf_path in REPORTS_DIR.glob("*.pdf"):
        meta = _parse_filename(pdf_path.name)
        if meta is None:
            continue
        entries.append(meta)

    entries.sort(key=lambda m: m["generated_at"], reverse=True)

    return [
        ReportResponse(
            report_id=m["report_id"],
            content="",
            zone_id=m["zone_id"],
            generated_at=m["generated_at"],
        )
        for m in entries[:20]
    ]


@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: str, request: Request  # pylint: disable=unused-argument
) -> FileResponse:
    """Download a completed report as a PDF file.

    Args:
        report_id: UUID of the report to download.
        request: FastAPI request (unused but required by FastAPI signature).

    Returns:
        PDF file as an attachment.

    Raises:
        HTTPException: 400 if report_id is not a valid UUID.
        HTTPException: 404 if the report file does not exist.
    """
    _validate_report_id(report_id)
    _ensure_reports_dir()
    pdf_path = _find_pdf(report_id)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail="Report not found.")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
        headers={"Content-Disposition": f'attachment; filename="{pdf_path.name}"'},
    )
