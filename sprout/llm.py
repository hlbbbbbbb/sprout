"""LLM Layer — thin wrapper over litellm for unified model access."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .types import CostReport, Message

logger = logging.getLogger(__name__)

# Lazy import litellm to allow importing the module without the dependency
_litellm = None


def _get_litellm():
    global _litellm
    if _litellm is None:
        try:
            import litellm

            litellm.set_verbose = False
            _litellm = litellm
        except ImportError:
            raise ImportError(
                "litellm is required. Install with: pip install litellm"
            )
    return _litellm


class LLMLayer:
    """Unified LLM interface using litellm.

    Supports Claude, GPT, DeepSeek, and any OpenAI-compatible API.
    Tracks token usage and cost for every call.
    """

    def __init__(
        self,
        model: str,
        fallback_model: str | None = None,
        temperature: float = 0.3,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.api_base = api_base
        self.api_key = api_key
        self.cost_report = CostReport()
        self._semaphore = asyncio.Semaphore(5)  # concurrent API calls (GLM-4.5 allows 10)

    async def chat(
        self,
        messages: list[Message],
        agent_id: str = "unknown",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Returns dict with keys: content, tool_calls, input_tokens, output_tokens, cost
        """
        litellm = _get_litellm()
        target_model = model or self.model
        temp = temperature if temperature is not None else self.temperature

        # Convert Message objects to dicts
        msg_dicts = []
        for m in messages:
            d = {"role": m.role, "content": m.content}
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            msg_dicts.append(d)

        kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": msg_dicts,
            "temperature": temp,
            "max_tokens": max_tokens,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if tools:
            kwargs["tools"] = tools

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                async with self._semaphore:
                    response = await litellm.acompletion(**kwargs)
                break
            except Exception as e:
                is_rate_limit = "RateLimitError" in type(e).__name__ or "rate" in str(e).lower()
                if is_rate_limit and attempt < max_retries:
                    wait = 2 ** attempt * 3  # 3s, 6s, 12s
                    logger.warning(f"LLM: rate limited, retry {attempt+1}/{max_retries} in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                if self.fallback_model and target_model != self.fallback_model:
                    logger.warning(f"LLM: {target_model} failed ({e}), falling back to {self.fallback_model}")
                    kwargs["model"] = self.fallback_model
                    async with self._semaphore:
                        response = await litellm.acompletion(**kwargs)
                    break
                raise

        # Extract response
        choice = response.choices[0]
        result = {
            "content": choice.message.content or "",
            "tool_calls": None,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "cost": (response._hidden_params.get("response_cost") or 0.0) if hasattr(response, "_hidden_params") else 0.0,
        }

        if choice.message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]

        # Track costs
        self.cost_report.add(
            node_id=agent_id,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost=result["cost"],
        )

        logger.debug(
            f"LLM: agent={agent_id} model={target_model} "
            f"in={result['input_tokens']} out={result['output_tokens']}"
        )

        return result

    def get_total_tokens(self) -> int:
        return self.cost_report.total_input_tokens + self.cost_report.total_output_tokens
