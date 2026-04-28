# Created by Metrum AI for AMD

"""Incident data investigator agent."""

try:
    from gaia.agents.base.tools import tool
except ImportError:  # pragma: no cover - exercised where GAIA is unavailable
    def tool(func):  # type: ignore[no-redef]
        """Fallback no-op decorator when GAIA is not installed."""
        return func

from smart_city.llm.agents.base import GaiaBaseAgent

_INVESTIGATOR_PROMPT = """\
You are InvestigatorAgent for a smart city operations center.

Goal:
- Gather incident evidence for the target zone and time window.
- Use tools to retrieve alerts, density trends, and recurring patterns.
- Produce a concise, compliance-ready evidence summary from retrieved data only.

Rules:
- Do not invent counts, times, locations, or severities.
- Call tools as needed before writing final summary.
- Use only values that are directly present in tool outputs.
- If a value is missing, write "UNKNOWN" (do not estimate).
- Keep output concise and factual.

Required output format (exact headings):
INCIDENT_FACTS
- zone_id: <value>
- time_window: <value>
- total_alerts: <integer or UNKNOWN>
- severity_observed: <CRITICAL/HIGH/MEDIUM/LOW/UNKNOWN>
- min_person_count: <integer or UNKNOWN>
- max_person_count: <integer or UNKNOWN>
- latest_alert_utc: <HH:MM UTC or UNKNOWN>

TIMELINE_EVIDENCE
- <HH:MM UTC> — <factual event from tool output>
- <HH:MM UTC> — <factual event from tool output>

TREND_EVIDENCE
- <single paragraph: density trend and anomalies from tool output only>

DATA_QUALITY
- missing_fields: <list or NONE>
- assumptions: NONE
"""


class InvestigatorAgent(GaiaBaseAgent):
    """Specialized agent for incident data retrieval and summarization."""

    def __init__(self, *, vllm_url: str, model: str, api_key: str) -> None:
        """Initialize investigator agent."""
        super().__init__(
            vllm_url=vllm_url,
            model=model,
            api_key=api_key,
            system_prompt=_INVESTIGATOR_PROMPT,
            agent_name="InvestigatorAgent",
            max_steps=8,
        )

    def _register_tools(self) -> None:
        """Register investigator toolset with GAIA."""

        @tool
        def get_zone_alerts(zone_id: str, hours: int) -> str:
            """Fetch recent alerts for a zone and lookback window."""
            return self._execute_tool_sync(
                "get_zone_alerts",
                {"zone_id": zone_id, "hours": hours},
            )

        @tool
        def get_density_trends(zone_id: str, hours: int) -> str:
            """Fetch hourly density aggregates for a zone."""
            return self._execute_tool_sync(
                "get_density_trends",
                {"zone_id": zone_id, "hours": hours},
            )

        @tool
        def get_historical_patterns(zone_id: str) -> str:
            """Fetch recurring historical crowd patterns for a zone."""
            return self._execute_tool_sync(
                "get_historical_patterns",
                {"zone_id": zone_id},
            )
