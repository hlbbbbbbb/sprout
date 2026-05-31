"""TaskTree — execution engine for the self-growing task tree.

Each node goes through:
  1. analyze() → should this task be split?
  2a. If no:  execute() → do the work
  2b. If yes: spawn children → children recurse through the same process
              → synthesize() children's results

This is recursive — children also analyze, and may split further.
The tree grows from the task structure, not from pre-designed topology.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .config import SproutConfig
from .llm import LLMLayer
from .types import (
    NodeStatus,
    TaskNode,
    TreeResult,
    WorkerResult,
)
from .worker import Worker

logger = logging.getLogger(__name__)


class TaskTree:
    """Grows and executes a tree of tasks.

    Usage:
        tree = TaskTree(config)
        result = await tree.run("Fix the bugs in auth.py and payment.py")
    """

    def __init__(self, config: SproutConfig | None = None):
        self.config = config or SproutConfig()
        self.llm = LLMLayer(
            model=self.config.model,
            fallback_model=self.config.fallback_model,
            temperature=self.config.temperature,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
        )
        self.nodes: dict[str, TaskNode] = {}
        self.events: list[str] = []

    async def run(self, task: str) -> TreeResult:
        """Run a task. The tree grows organically from here."""
        start_time = time.time()

        root = self._create_node(task=task, context="", parent_id=None, depth=0)
        logger.info(f"TaskTree: root {root.id}")

        root_result = await self._execute_node(root.id)

        elapsed = time.time() - start_time
        answer = ""
        if root_result:
            answer = root_result.content
        else:
            answer = self._gather_from_children(root.id)

        return TreeResult(
            answer=answer,
            tree_summary=self._format_tree(),
            total_nodes=len(self.nodes),
            max_depth=max(n.depth for n in self.nodes.values()),
            total_spawns=sum(len(n.children_ids) for n in self.nodes.values()),
            cost_report=self.llm.cost_report,
            elapsed_seconds=elapsed,
            events=self.events,
        )

    async def _execute_node(self, node_id: str) -> WorkerResult | None:
        """Execute a node: analyze → (split or execute) → return result."""
        node = self.nodes[node_id]
        node.status = NodeStatus.RUNNING
        node.started_at = time.time()

        logger.info(f"{'  ' * node.depth}▶ [{node.id}] {node.task[:70]}")

        # Budget check
        if self.llm.get_total_tokens() >= self.config.max_total_tokens:
            logger.warning(f"[{node.id}] token budget exhausted")
            node.status = NodeStatus.FAILED
            self.events.append(f"BUDGET: {node.id}")
            return WorkerResult(content="(token budget exhausted)")

        worker = Worker(self.config, self.llm)

        # Can this node split?
        can_split = (
            node.depth < self.config.max_depth - 1
            and len(self.nodes) < self.config.max_total_nodes
        )

        # ── Phase 1: Analyze ──
        spawn_requests = []
        if can_split:
            spawn_requests = await worker.analyze(node.id, node.task, node.context)
            # Enforce limits
            spawn_requests = spawn_requests[:self.config.max_children_per_node]
            remaining = self.config.max_total_nodes - len(self.nodes)
            spawn_requests = spawn_requests[:remaining]

        # ── Phase 2a: No split → Execute directly ──
        if not spawn_requests:
            result = await worker.execute(node.id, node.task, node.context, node.approach)
            if result:
                node.result = result
                node.status = NodeStatus.COMPLETED
            else:
                node.status = NodeStatus.ABANDONED
                self.events.append(f"ABANDON: {node.id}")
            node.completed_at = time.time()
            self._log_done(node)
            return result

        # ── Phase 2b: Split → Run children → Synthesize ──
        node.status = NodeStatus.WAITING

        # Create children
        children = []
        for req in spawn_requests:
            child = self._create_node(
                task=req.task,
                context=req.context,
                approach=req.approach,
                parent_id=node.id,
                depth=node.depth + 1,
            )
            node.children_ids.append(child.id)
            children.append(child)
            self.events.append(f"SPAWN: {child.id} ← {node.id} ({req.reason})")
            logger.info(f"{'  ' * node.depth}  ↳ spawn {child.id}: {req.task[:55]}")

        # Execute children in parallel with straggler detection
        children_results = await self._execute_children_with_straggler_detection(
            node, children, worker,
        )

        # Synthesize
        if children_results:
            result = await worker.synthesize(node.id, node.task, children_results)
        else:
            result = None

        if result:
            node.result = result
            node.status = NodeStatus.COMPLETED
        else:
            # Fallback: gather whatever we got
            gathered = self._gather_from_children(node.id)
            if gathered:
                result = WorkerResult(content=gathered)
                node.result = result
            node.status = NodeStatus.COMPLETED

        node.completed_at = time.time()
        self._log_done(node)
        return result

    async def _execute_children_with_straggler_detection(
        self,
        parent: TaskNode,
        children: list[TaskNode],
        worker: Worker,
    ) -> list[WorkerResult]:
        """Execute children with straggler detection.

        Uses asyncio tasks + periodic check instead of gather.
        When a child takes much longer than its siblings,
        cancel it → re-analyze → split into smaller pieces.
        """
        # Create async tasks
        task_map: dict[asyncio.Task, TaskNode] = {}
        for child in children:
            t = asyncio.create_task(self._execute_node(child.id))
            task_map[t] = child

        completed_results: list[WorkerResult] = []
        completed_times: list[float] = []
        pending_tasks = set(task_map.keys())

        while pending_tasks:
            # Wait for the next child to complete, with periodic straggler checks
            done, pending_tasks = await asyncio.wait(
                pending_tasks,
                timeout=10.0,  # check every 10s
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Process newly completed tasks
            for t in done:
                child = task_map[t]
                try:
                    res = t.result()
                    if res is not None:
                        completed_results.append(res)
                        elapsed = (child.completed_at or time.time()) - (child.started_at or time.time())
                        completed_times.append(elapsed)
                except Exception as e:
                    logger.error(f"[{child.id}] failed: {e}")
                    child.status = NodeStatus.FAILED
                    self.events.append(f"FAIL: {child.id}")

            # Check for stragglers among still-pending tasks
            if not pending_tasks or len(completed_times) < self.config.straggler_min_siblings:
                continue

            avg_time = sum(completed_times) / len(completed_times)
            threshold = avg_time * self.config.straggler_multiplier
            now = time.time()

            for t in list(pending_tasks):
                child = task_map[t]
                child_elapsed = now - (child.started_at or now)

                if child_elapsed <= threshold:
                    continue

                # Can we re-split this child?
                can_resplit = (
                    child.depth < self.config.max_depth - 1
                    and len(self.nodes) + 2 <= self.config.max_total_nodes
                )

                if not can_resplit:
                    continue  # let it finish naturally

                # ── Straggler detected → cancel & re-split ──
                logger.warning(
                    f"{'  ' * child.depth}⚡ STRAGGLER [{child.id}] "
                    f"{child_elapsed:.0f}s > threshold {threshold:.0f}s — re-splitting"
                )
                self.events.append(
                    f"STRAGGLER: {child.id} ({child_elapsed:.0f}s > {threshold:.0f}s)"
                )

                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                pending_tasks.discard(t)

                # Re-analyze the straggler's task
                resplit_worker = Worker(self.config, self.llm)
                sub_requests = await resplit_worker.analyze(
                    child.id, child.task, child.context,
                )

                if len(sub_requests) >= 2:
                    # Split worked → create grandchildren
                    sub_requests = sub_requests[:self.config.max_children_per_node]
                    remaining = self.config.max_total_nodes - len(self.nodes)
                    sub_requests = sub_requests[:remaining]

                    child.status = NodeStatus.WAITING
                    sub_children = []
                    for req in sub_requests:
                        gc = self._create_node(
                            task=req.task,
                            context=req.context,
                            approach=req.approach,
                            parent_id=child.id,
                            depth=child.depth + 1,
                        )
                        child.children_ids.append(gc.id)
                        sub_children.append(gc)
                        self.events.append(
                            f"RESPLIT: {gc.id} ← {child.id} ({req.reason})"
                        )
                        logger.info(
                            f"{'  ' * child.depth}  ↳ resplit {gc.id}: {req.task[:55]}"
                        )

                    # Run grandchildren
                    sub_results = await self._execute_children_with_straggler_detection(
                        child, sub_children, resplit_worker,
                    )

                    # Synthesize grandchildren results into child result
                    if sub_results:
                        child_result = await resplit_worker.synthesize(
                            child.id, child.task, sub_results,
                        )
                        if child_result:
                            child.result = child_result
                            child.status = NodeStatus.COMPLETED
                            child.completed_at = time.time()
                            completed_results.append(child_result)
                            elapsed = (child.completed_at or 0) - (child.started_at or 0)
                            completed_times.append(elapsed)
                            self._log_done(child)
                    else:
                        child.status = NodeStatus.FAILED
                        self.events.append(f"FAIL: {child.id}")
                else:
                    # Re-split didn't work → run as single task
                    logger.info(
                        f"{'  ' * child.depth}  ↳ resplit failed, re-executing [{child.id}]"
                    )
                    child.started_at = time.time()
                    res = await resplit_worker.execute(
                        child.id, child.task, child.context, child.approach,
                    )
                    if res:
                        child.result = res
                        child.status = NodeStatus.COMPLETED
                        child.completed_at = time.time()
                        completed_results.append(res)
                        elapsed = (child.completed_at or 0) - (child.started_at or 0)
                        completed_times.append(elapsed)
                        self._log_done(child)
                    else:
                        child.status = NodeStatus.FAILED
                        self.events.append(f"FAIL: {child.id}")

        return completed_results

    def _create_node(self, task: str, context: str, parent_id: str | None, depth: int, approach: str = "") -> TaskNode:
        node = TaskNode(task=task, context=context, approach=approach, parent_id=parent_id, depth=depth)
        self.nodes[node.id] = node
        return node

    def _gather_from_children(self, node_id: str) -> str:
        node = self.nodes[node_id]
        parts = []
        for cid in node.children_ids:
            child = self.nodes[cid]
            if child.result:
                parts.append(f"## {child.task}\n\n{child.result.content}")
        return "\n\n---\n\n".join(parts) if parts else ""

    def _log_done(self, node: TaskNode):
        elapsed = (node.completed_at or 0) - (node.started_at or 0)
        icon = "✓" if node.status == NodeStatus.COMPLETED else "✗"
        rtype = node.result.result_type if node.result else "-"
        logger.info(f"{'  ' * node.depth}{icon} [{node.id}] {rtype} ({elapsed:.1f}s)")

    def _format_tree(self, node_id: str | None = None, indent: int = 0) -> str:
        if node_id is None:
            roots = [n for n in self.nodes.values() if n.parent_id is None]
            if not roots:
                return "(empty tree)"
            return self._format_tree(roots[0].id, 0)

        node = self.nodes[node_id]
        icons = {
            NodeStatus.COMPLETED: "✓", NodeStatus.FAILED: "✗",
            NodeStatus.ABANDONED: "⊘", NodeStatus.RUNNING: "▶",
            NodeStatus.WAITING: "⏳", NodeStatus.PENDING: "○",
        }
        icon = icons.get(node.status, "?")
        tokens = ""
        if node.id in self.llm.cost_report.by_node:
            s = self.llm.cost_report.by_node[node.id]
            tokens = f" [{s['input']+s['output']}tok]"

        line = f"{'  ' * indent}{icon} {node.id}: {node.task[:60]}{tokens}\n"
        for cid in node.children_ids:
            line += self._format_tree(cid, indent + 1)
        return line
