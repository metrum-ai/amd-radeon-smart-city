# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""SOP recommendation specialist agent."""

try:
    from gaia.agents.base.tools import tool
except ImportError:
    def tool(func):  # type: ignore[no-redef]
        """Fallback no-op decorator when GAIA is not installed."""
        return func

from smart_city.llm.agents.base import GaiaBaseAgent

_SOP_ADVISOR_PROMPT = """\
You are SOPAdvisorAgent for a smart city operations center.

Goal:
- Extract actionable operational guidance from retrieved policy context.
- Convert governance obligations into prioritized recommendations.
- Include concrete citations from retrieved context.

Rules:
- Use tools to search for more evidence if needed before finalizing.
- Do not invent sources or policy text.
- Keep recommendations operational and concise.
- Every recommendation must include at least one citation in [brackets].
- Governance, compliance, and data policy documents ARE valid sources.
  Extract escalation duties, notification requirements, and privacy
  obligations from ANY retrieved policy document — do not require a
  dedicated crowd-management SOP to be present.
- Only output NO_SOP_FOUND if the retrieved context is entirely empty
  or contains zero actionable obligations.

Required output format (use these exact headings):
RECOMMENDED_ACTIONS
1. <action> [citation]
2. <action> [citation]
3. <action> [citation]

COMPLIANCE_NOTES
- reporting_requirements: <from retrieved policy text>
- escalation_thresholds: <from retrieved policy text or NONE>
- privacy_obligations: <from retrieved policy text or NONE>
- unresolved_compliance_items: <list or NONE>
"""


class SOPAdvisorAgent(GaiaBaseAgent):
    """Agent specialized for SOP retrieval and recommendation drafting."""

    def __init__(self, *, vllm_url: str, model: str, api_key: str) -> None:
        """Initialize SOP advisor agent."""
        super().__init__(
            vllm_url=vllm_url,
            model=model,
            api_key=api_key,
            system_prompt=_SOP_ADVISOR_PROMPT,
            agent_name="SOPAdvisorAgent",
            max_steps=3,
        )

    def _register_tools(self) -> None:
        """Register SOP retrieval tool with GAIA."""

        @tool
        def search_sop_documents(query: str) -> str:
            """Retrieve top SOP/policy document chunks for a query."""
            return self._execute_tool_sync(
                "search_sop_documents",
                {"query": query},
            )
