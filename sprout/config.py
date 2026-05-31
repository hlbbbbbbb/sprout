"""Sprout v2 configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SproutConfig:
    """Top-level configuration for a Sprout run.

    The key constraints are token budget and tree shape limits.
    Everything else is left to the agent's judgment.
    """

    # Model
    model: str = "openai/glm-4-flash"
    fallback_model: str | None = None
    temperature: float = 0.3
    api_base: str | None = "https://open.bigmodel.cn/api/paas/v4"
    api_key: str | None = None  # set via env or here

    # Budget — the hard constraint that prevents infinite growth
    max_total_tokens: int = 500_000

    # Tree shape limits — prevent exponential explosion
    max_children_per_node: int = 3    # each agent can spawn at most 3 subtasks
    max_depth: int = 4                # tree can't grow deeper than 4 levels
    max_total_nodes: int = 15         # total nodes across the entire tree

    # Worker
    max_tokens_per_call: int = 4096   # max output tokens per LLM call
    max_retries: int = 1              # retry on LLM failure

    # Straggler mitigation — re-split slow children
    straggler_multiplier: float = 2.5  # child taking > avg * this → straggler
    straggler_min_siblings: int = 2    # need at least N completed siblings to judge

    # Output
    verbose: bool = True
