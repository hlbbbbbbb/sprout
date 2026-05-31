"""Core type definitions for Sprout v2 — tree-based self-growing agents."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeStatus(str, Enum):
    """Status of a task node in the tree."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"      # waiting for children to finish
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"  # agent decided this subtask isn't useful


@dataclass
class SubtaskRequest:
    """A request from a worker to spawn a child subtask.

    Workers produce these by calling the spawn_subtask tool.
    The framework turns each one into a new TaskNode.
    """
    task: str
    context: str = ""           # what the child needs to know
    reason: str = ""            # why spawn this
    approach: str = ""          # HOW to do it: methodology, output format, constraints
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class WorkerResult:
    """The output of a worker after it finishes.

    Workers produce this by calling the report_result tool.
    """
    content: str
    result_type: str = "general"   # "code", "analysis", "fix", "test", etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskNode:
    """A node in the task tree.

    Each node = one unit of work. A node can have children (subtasks).
    The tree grows when workers call spawn_subtask.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    task: str = ""
    context: str = ""           # inherited from parent + parent's additions
    approach: str = ""          # HOW to do this: methodology, output format, role
    status: NodeStatus = NodeStatus.PENDING
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0

    # Filled after execution
    result: WorkerResult | None = None
    spawn_requests: list[SubtaskRequest] = field(default_factory=list)

    # Metrics
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class Message:
    """A chat message."""
    role: str         # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


@dataclass
class CostReport:
    """Token and cost tracking."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    by_node: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, node_id: str, input_tokens: int, output_tokens: int, cost: float):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        if node_id not in self.by_node:
            self.by_node[node_id] = {"input": 0, "output": 0, "cost": 0.0}
        self.by_node[node_id]["input"] += input_tokens
        self.by_node[node_id]["output"] += output_tokens
        self.by_node[node_id]["cost"] += cost


@dataclass
class TreeResult:
    """The final output of a Sprout run."""
    answer: str
    tree_summary: str       # visual representation of the task tree
    total_nodes: int = 0
    max_depth: int = 0
    total_spawns: int = 0   # how many times agents spawned children
    cost_report: CostReport | None = None
    elapsed_seconds: float = 0.0
    events: list[str] = field(default_factory=list)  # lifecycle event log
