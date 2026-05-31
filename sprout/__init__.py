"""Sprout — Self-growing task tree for LLM agents.

Agents decide when to split, merge, and die. No pre-designed topology.

Usage:
    import asyncio
    from sprout import TaskTree

    result = asyncio.run(TaskTree().run("Your task here"))
    print(result.answer)
"""

from .config import SproutConfig
from .task_tree import TaskTree
from .types import TreeResult

__all__ = ["TaskTree", "SproutConfig", "TreeResult"]
__version__ = "0.2.0"
