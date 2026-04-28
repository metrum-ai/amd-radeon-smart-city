# Created by Metrum AI for AMD

"""GAIA-backed base agent runtime for incident workflows."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from gaia.agents.base.agent import Agent
    from gaia.agents.base.tools import _TOOL_REGISTRY
except ImportError as exc:  # pragma: no cover - exercised in environments without GAIA
    Agent = object  # type: ignore[assignment,misc]
    _TOOL_REGISTRY = {}
    _GAIA_IMPORT_ERROR: Optional[ImportError] = exc
else:
    _GAIA_IMPORT_ERROR = None
    # Guard against lemonade occasionally returning choices=None.
    #
    # Lemonade has an 8-thread HTTP pool (server.cpp ThreadPool(8)) so it
    # queues concurrent requests internally — no semaphore is needed.  The
    # real issue is that when lemonade returns an error body instead of a
    # normal completion, the openai SDK sets choices=None, and the stock
    # provider code does response.choices[0] without any guard.
    #
    # Fix: retry up to 3 times with linear back-off before re-raising so
    # that transient lemonade hiccups don't abort the agent loop.
    try:
        import time as _time

        from gaia.llm.providers import openai_provider as _oai_mod

        def _safe_chat(self, messages, model=None, stream=False, **kwargs):
            """Retry-wrapped chat that guards against choices=None."""
            for _attempt in range(3):
                response = self._client.chat.completions.create(
                    model=model or self._model,
                    messages=messages,
                    stream=stream,
                    **kwargs,
                )
                if stream:
                    return self._handle_stream(response)
                if response and response.choices:
                    content = response.choices[0].message.content
                    return content if content is not None else ""
                # choices is None/empty — transient lemonade error, retry
                logger.warning(
                    "LLM returned no choices (attempt %d/3); retrying…",
                    _attempt + 1,
                )
                if _attempt < 2:
                    _time.sleep(2 * (_attempt + 1))  # 2 s, 4 s
            raise ValueError(
                "LLM returned no choices after 3 attempts "
                "(lemonade may have returned an error response)."
            )

        _oai_mod.OpenAIProvider.chat = _safe_chat
        logger.debug("Patched OpenAIProvider.chat with retry+None-choices guard.")
    except Exception as _patch_exc:
        logger.warning("Could not patch OpenAIProvider.chat: %s", _patch_exc)

from smart_city.llm.agents.tools import execute_tool


@dataclass
class TraceStep:
    """Single agent trace step for UI visibility."""

    agent: str
    tool: Optional[str]
    summary: str
    timestamp: datetime
    raw_result: Optional[str] = None  # full tool output (not truncated)


class GaiaBaseAgent(Agent):
    """GAIA Agent wrapper with async bridge for FastAPI services."""

    def __init__(
        self,
        *,
        vllm_url: str,
        model: str,
        api_key: str,
        agent_name: str,
        system_prompt: str,
        max_steps: int,
    ) -> None:
        """Initialize GAIA runtime against OpenAI-compatible vLLM."""
        if _GAIA_IMPORT_ERROR is not None:
            raise RuntimeError(
                "amd-gaia is required for GAIA-backed agents."
            ) from _GAIA_IMPORT_ERROR

        cleaned_url = vllm_url.strip().rstrip("/")
        base_url = cleaned_url[:-3] if cleaned_url.endswith("/v1") else cleaned_url

        self._agent_name = agent_name
        self._system_prompt_text = system_prompt
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._db_pool: Optional[Any] = None
        self._vector_store: Optional[Any] = None
        self._embedder: Optional[Any] = None
        self._from_date: Optional[str] = None
        self._to_date: Optional[str] = None
        # Called synchronously from the worker thread after each tool call.
        # Signature: Callable[[TraceStep], None]
        self._step_callback: Optional[Any] = None

        # GAIA's OpenAI provider reads env vars, so set them explicitly.
        os.environ["OPENAI_BASE_URL"] = f"{base_url}/v1"
        os.environ["OPENAI_API_KEY"] = api_key or "sk-local"
        _TOOL_REGISTRY.clear()

        super().__init__(
            use_chatgpt=True,
            model_id=model,
            max_steps=max_steps,
            skip_lemonade=True,
            silent_mode=True,
        )

        # Lemonade natively separates thinking tokens into `reasoning_content`
        # and returns clean text in `message.content`, so no patching is needed
        # to strip <think> blocks.  The GAIA OpenAI provider already reads only
        # `message.content`, which is already clean when served by lemonade.
        #
        # If the agent's final answer still cannot be parsed (model does not
        # follow INCIDENT_FACTS format), the Orchestrator falls back to
        # reconstructing the summary directly from tool call results in the
        # trace — see `_build_summary_from_trace`.
        logger.debug(
            "%s: GAIA agent initialised (thinking enabled via lemonade "
            "reasoning_content separation).",
            agent_name,
        )

    def _get_system_prompt(self) -> str:
        """Return the agent-specific prompt for GAIA."""
        return self._system_prompt_text

    async def run(
        self,
        user_message: str,
        *,
        db_pool: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        embedder: Optional[Any] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        step_callback: Optional[Any] = None,
    ) -> tuple[str, list[TraceStep]]:
        """Run synchronous GAIA loop in a worker thread.

        Args:
            step_callback: Optional ``Callable[[TraceStep], None]`` called
                from the worker thread immediately after each tool completes.
                Must be thread-safe (no asyncio primitives).
        """
        self._loop = asyncio.get_running_loop()
        self._db_pool = db_pool
        self._vector_store = vector_store
        self._embedder = embedder
        self._from_date = from_date
        self._to_date = to_date
        self._step_callback = step_callback

        # Re-register this agent's tools so the global registry reflects only
        # the current agent's toolset (each init clears the registry).
        _TOOL_REGISTRY.clear()
        self._register_tools()

        try:
            result = await asyncio.to_thread(self.process_query, user_message)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "%s process_query raised: %s", self._agent_name, exc
            )
            return "", []

        if not isinstance(result, dict):
            return "", []
        final_text = str(result.get("result") or "").strip()
        return final_text, self._extract_trace(result)

    def _execute_tool_sync(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute async tool handler from GAIA worker thread.

        Injects from_ts / to_ts into args when the agent was given an
        absolute date range so that SQL queries target the correct window
        rather than a rolling offset from NOW().
        """
        if self._loop is None:
            return json.dumps({"error": "Runtime loop unavailable"})

        enriched = dict(args)
        if self._from_date:
            enriched.setdefault("from_ts", self._from_date)
        if self._to_date:
            enriched.setdefault("to_ts", self._to_date)

        future = asyncio.run_coroutine_threadsafe(
            execute_tool(
                tool_name=tool_name,
                args=enriched,
                db_pool=self._db_pool,
                vector_store=self._vector_store,
                embedder=self._embedder,
            ),
            self._loop,
        )
        result = future.result(timeout=120.0)

        # Fire live per-step callback so the trace updates in real time
        # rather than batching after the full agent run.  Called from the
        # worker thread — the callback must be synchronous and thread-safe.
        if self._step_callback is not None:
            try:
                self._step_callback(
                    TraceStep(
                        agent=self._agent_name,
                        tool=tool_name,
                        summary=self._summarize_result(result),
                        timestamp=datetime.now(timezone.utc),
                        raw_result=result,
                    )
                )
            except Exception:  # pylint: disable=broad-except
                pass  # never let a callback error abort a tool call

        return result

    def _extract_trace(self, result: dict[str, Any]) -> list[TraceStep]:
        """Translate GAIA conversation events into UI trace steps.

        Captures both tool-call results (role=tool) and the agent's final
        answer (role=assistant) so the handoff chain is always visible even
        when no tools were invoked.
        """
        conversation = result.get("conversation", [])
        trace: list[TraceStep] = []
        for entry in conversation:
            role = entry.get("role", "")
            if role == "tool":
                tool_name = str(entry.get("name") or "")
                raw_content = entry.get("content", "")
                full_result = (
                    raw_content
                    if isinstance(raw_content, str)
                    else json.dumps(raw_content)
                )
                summary = self._summarize_result(full_result)
                trace.append(
                    TraceStep(
                        agent=self._agent_name,
                        tool=tool_name,
                        summary=summary,
                        timestamp=datetime.now(timezone.utc),
                        raw_result=full_result,
                    )
                )
            elif role == "assistant":
                raw_content = entry.get("content", "")
                if not raw_content:
                    continue
                thought = self._extract_thought(
                    raw_content
                    if isinstance(raw_content, str)
                    else json.dumps(raw_content)
                )
                if not thought:
                    continue
                trace.append(
                    TraceStep(
                        agent=self._agent_name,
                        tool=None,
                        summary=thought,
                        timestamp=datetime.now(timezone.utc),
                    )
                )
        # If GAIA returned no conversation entries at all, emit a single
        # fallback step so the agent is always visible in the trace.
        if not trace:
            final_text = str(result.get("result") or "").strip()
            trace.append(
                TraceStep(
                    agent=self._agent_name,
                    tool=None,
                    summary=(
                        self._summarize_result(final_text)
                        if final_text
                        else f"{self._agent_name} completed (no tool calls)."
                    ),
                    timestamp=datetime.now(timezone.utc),
                )
            )
        return trace

    @staticmethod
    def _extract_thought(raw: str) -> str:
        """Extract the agent's reasoning thought from a JSON response.

        GAIA assistant messages are JSON objects with a ``thought`` field.
        Returns the thought string when present, otherwise falls back to
        the full raw text so nothing is silently dropped.

        Args:
            raw: Raw assistant content string.

        Returns:
            Human-readable thought string, or empty string to skip the step.
        """
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                thought = str(obj.get("thought") or "").strip()
                if thought:
                    return thought
                # If there is no thought, skip intermediate planning steps
                # that carry no user-visible information.
                return ""
        except (json.JSONDecodeError, ValueError):
            pass
        # Plain text response — return as-is (truncated for display)
        one_line = " ".join(raw.split())
        return one_line[:300] if len(one_line) > 300 else one_line

    @staticmethod
    def _summarize_result(result: str) -> str:
        """Create a short single-line summary for trace output."""
        one_line = " ".join(result.split())
        if len(one_line) <= 180:
            return one_line
        return f"{one_line[:177]}..."
