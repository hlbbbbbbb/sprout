"""Tests for the task tree execution engine (two-phase flow)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from sprout.config import SproutConfig
from sprout.task_tree import TaskTree
from sprout.types import NodeStatus, SubtaskRequest, WorkerResult
from sprout.worker import Worker


@pytest.fixture
def config():
    return SproutConfig(
        model="test-model",
        max_total_tokens=100_000,
        max_children_per_node=3,
        max_depth=3,
        max_total_nodes=10,
    )


class TestSimpleExecution:
    """Tests where analyze() returns no split → execute() directly."""

    @pytest.mark.asyncio
    async def test_single_node_no_split(self, config):
        """analyze returns [] → execute runs → single node tree."""
        tree = TaskTree(config)

        with patch.object(Worker, "analyze", new_callable=AsyncMock) as mock_analyze, \
             patch.object(Worker, "execute", new_callable=AsyncMock) as mock_execute:
            mock_analyze.return_value = []
            mock_execute.return_value = WorkerResult(content="Fixed with bcrypt")

            result = await tree.run("Fix the auth bug")

        assert "bcrypt" in result.answer
        assert result.total_nodes == 1
        assert result.max_depth == 0
        assert result.total_spawns == 0

    @pytest.mark.asyncio
    async def test_abandoned_task(self, config):
        """execute returns None → node marked as abandoned."""
        tree = TaskTree(config)

        with patch.object(Worker, "analyze", new_callable=AsyncMock) as mock_analyze, \
             patch.object(Worker, "execute", new_callable=AsyncMock) as mock_execute:
            mock_analyze.return_value = []
            mock_execute.return_value = None

            await tree.run("Fix something vague")

        root = list(tree.nodes.values())[0]
        assert root.status == NodeStatus.ABANDONED

    @pytest.mark.asyncio
    async def test_execute_not_called_when_split(self, config):
        """When analyze returns subtasks, execute should NOT be called."""
        tree = TaskTree(config)

        with patch.object(Worker, "analyze", new_callable=AsyncMock) as mock_analyze, \
             patch.object(Worker, "execute", new_callable=AsyncMock) as mock_execute, \
             patch.object(Worker, "synthesize", new_callable=AsyncMock) as mock_synth:

            call_count = {"analyze": 0}

            async def analyze_side_effect(node_id, task, context=""):
                call_count["analyze"] += 1
                if call_count["analyze"] == 1:
                    # Root splits
                    return [
                        SubtaskRequest(task="Sub A", reason="independent"),
                        SubtaskRequest(task="Sub B", reason="independent"),
                    ]
                return []  # Children don't split

            mock_analyze.side_effect = analyze_side_effect
            mock_execute.return_value = WorkerResult(content="Done")
            mock_synth.return_value = WorkerResult(content="Combined")

            await tree.run("Complex task")

        # execute should be called for children (2 times), NOT for root
        assert mock_execute.call_count == 2


class TestSpawning:
    """Tests where analyze() returns subtasks → children are spawned."""

    @pytest.mark.asyncio
    async def test_spawn_creates_children(self, config):
        """analyze returns 2 subtasks → tree has 3 nodes (root + 2 children)."""
        tree = TaskTree(config)

        call_count = {"analyze": 0}

        async def analyze_side_effect(node_id, task, context=""):
            call_count["analyze"] += 1
            if call_count["analyze"] == 1:
                return [
                    SubtaskRequest(task="Fix auth.py", reason="Independent module"),
                    SubtaskRequest(task="Fix cart.py", reason="Independent module"),
                ]
            return []

        with patch.object(Worker, "analyze", new_callable=AsyncMock, side_effect=analyze_side_effect), \
             patch.object(Worker, "execute", new_callable=AsyncMock) as mock_execute, \
             patch.object(Worker, "synthesize", new_callable=AsyncMock) as mock_synth:
            mock_execute.return_value = WorkerResult(content="Fixed")
            mock_synth.return_value = WorkerResult(content="All modules fixed")

            result = await tree.run("Fix all bugs")

        assert result.total_nodes == 3
        assert result.total_spawns == 2
        assert "All modules fixed" in result.answer

    @pytest.mark.asyncio
    async def test_max_depth_prevents_deep_spawning(self, config):
        """At max depth, can_split is False → analyze is skipped."""
        config.max_depth = 2
        tree = TaskTree(config)

        call_count = {"analyze": 0}

        async def analyze_side_effect(node_id, task, context=""):
            call_count["analyze"] += 1
            # Always try to split
            return [
                SubtaskRequest(task="Sub A", reason="test"),
                SubtaskRequest(task="Sub B", reason="test"),
            ]

        with patch.object(Worker, "analyze", new_callable=AsyncMock, side_effect=analyze_side_effect), \
             patch.object(Worker, "execute", new_callable=AsyncMock) as mock_execute, \
             patch.object(Worker, "synthesize", new_callable=AsyncMock) as mock_synth:
            mock_execute.return_value = WorkerResult(content="Done")
            mock_synth.return_value = WorkerResult(content="Combined")

            result = await tree.run("Deep task")

        # Depth 0 splits → depth 1 children. max_depth=2, so depth 1 = max_depth-1 → can_split=False
        # So analyze should only be called once (for root at depth 0)
        assert result.max_depth <= config.max_depth - 1

    @pytest.mark.asyncio
    async def test_max_children_per_node(self, config):
        """analyze returns 5 subtasks but max_children_per_node=2 → only 2 created."""
        config.max_children_per_node = 2
        tree = TaskTree(config)

        call_count = {"analyze": 0}

        async def analyze_side_effect(node_id, task, context=""):
            call_count["analyze"] += 1
            if call_count["analyze"] == 1:
                return [
                    SubtaskRequest(task=f"Subtask {i}", reason="test")
                    for i in range(5)
                ]
            return []

        with patch.object(Worker, "analyze", new_callable=AsyncMock, side_effect=analyze_side_effect), \
             patch.object(Worker, "execute", new_callable=AsyncMock) as mock_execute, \
             patch.object(Worker, "synthesize", new_callable=AsyncMock) as mock_synth:
            mock_execute.return_value = WorkerResult(content="Done")
            mock_synth.return_value = WorkerResult(content="Combined")

            await tree.run("Many subtasks")

        root = [n for n in tree.nodes.values() if n.parent_id is None][0]
        assert len(root.children_ids) <= 2


class TestTreeStructure:
    """Tests for tree formatting and structure."""

    @pytest.mark.asyncio
    async def test_tree_summary_format(self, config):
        """Tree summary should show the node hierarchy."""
        tree = TaskTree(config)

        with patch.object(Worker, "analyze", new_callable=AsyncMock) as mock_analyze, \
             patch.object(Worker, "execute", new_callable=AsyncMock) as mock_execute:
            mock_analyze.return_value = []
            mock_execute.return_value = WorkerResult(content="Done")

            result = await tree.run("Simple task")

        assert "✓" in result.tree_summary
        assert "Simple task" in result.tree_summary

    @pytest.mark.asyncio
    async def test_events_track_spawns(self, config):
        """Events list should track spawn operations."""
        tree = TaskTree(config)

        call_count = {"analyze": 0}

        async def analyze_side_effect(node_id, task, context=""):
            call_count["analyze"] += 1
            if call_count["analyze"] == 1:
                return [
                    SubtaskRequest(task="Sub A", reason="reason A"),
                    SubtaskRequest(task="Sub B", reason="reason B"),
                ]
            return []

        with patch.object(Worker, "analyze", new_callable=AsyncMock, side_effect=analyze_side_effect), \
             patch.object(Worker, "execute", new_callable=AsyncMock) as mock_execute, \
             patch.object(Worker, "synthesize", new_callable=AsyncMock) as mock_synth:
            mock_execute.return_value = WorkerResult(content="Done")
            mock_synth.return_value = WorkerResult(content="Combined")

            result = await tree.run("Task")

        spawn_events = [e for e in result.events if e.startswith("SPAWN")]
        assert len(spawn_events) == 2


class TestStragglerDetection:
    """Tests for straggler detection and re-splitting."""

    @pytest.mark.asyncio
    async def test_straggler_triggers_resplit(self, config):
        """A slow child gets cancelled and re-split into smaller pieces."""
        config.straggler_multiplier = 1.5
        config.straggler_min_siblings = 2
        config.max_depth = 4
        config.max_total_nodes = 20
        tree = TaskTree(config)

        # Track which tasks have been analyzed to distinguish first vs re-analyze
        analyzed_tasks = set()

        async def analyze_side_effect(node_id, task, context=""):
            if task not in analyzed_tasks and node_id == list(tree.nodes.keys())[0]:
                # Root: split into 3
                analyzed_tasks.add(task)
                return [
                    SubtaskRequest(task="Fast A", reason="independent"),
                    SubtaskRequest(task="Fast B", reason="independent"),
                    SubtaskRequest(task="Slow C", reason="independent"),
                ]
            elif task == "Slow C" and task in analyzed_tasks:
                # Re-analyze after straggler detection → NOW split
                return [
                    SubtaskRequest(task="Slow C part 1", reason="split slow task"),
                    SubtaskRequest(task="Slow C part 2", reason="split slow task"),
                ]
            # First analyze of "Slow C" or any other task → don't split
            analyzed_tasks.add(task)
            return []

        async def execute_side_effect(node_id, task, context="", approach=""):
            if task == "Slow C":
                # Very slow — will be cancelled by straggler detection
                await asyncio.sleep(300)
                return WorkerResult(content="Slow done")
            await asyncio.sleep(0.01)
            return WorkerResult(content=f"Done: {task}", result_type="code")

        with patch.object(Worker, "analyze", new_callable=AsyncMock, side_effect=analyze_side_effect), \
             patch.object(Worker, "execute", new_callable=AsyncMock, side_effect=execute_side_effect), \
             patch.object(Worker, "synthesize", new_callable=AsyncMock) as mock_synth:
            mock_synth.return_value = WorkerResult(content="All combined")

            result = await tree.run("Complex task")

        straggler_events = [e for e in result.events if "STRAGGLER" in e]
        resplit_events = [e for e in result.events if "RESPLIT" in e]
        assert len(straggler_events) >= 1
        assert len(resplit_events) >= 2  # Slow C → 2 sub-tasks
        assert result.total_nodes >= 5  # root + 3 children + 2 grandchildren

    @pytest.mark.asyncio
    async def test_straggler_fallback_when_resplit_fails(self, config):
        """If re-split returns no subtasks, re-execute the task directly."""
        config.straggler_multiplier = 1.5
        config.straggler_min_siblings = 2
        config.max_depth = 4
        tree = TaskTree(config)

        analyzed_tasks = set()
        execute_count = {"Slow C": 0}

        async def analyze_side_effect(node_id, task, context=""):
            if task not in analyzed_tasks and not analyzed_tasks:
                analyzed_tasks.add(task)
                return [
                    SubtaskRequest(task="Fast A", reason="independent"),
                    SubtaskRequest(task="Fast B", reason="independent"),
                    SubtaskRequest(task="Slow C", reason="independent"),
                ]
            analyzed_tasks.add(task)
            # All re-analyze attempts return no split
            return []

        async def execute_side_effect(node_id, task, context="", approach=""):
            if task == "Slow C":
                execute_count["Slow C"] += 1
                if execute_count["Slow C"] == 1:
                    # First time: very slow → will be cancelled
                    await asyncio.sleep(300)
                    return WorkerResult(content="Slow done")
                # Re-execution after straggler: fast
                return WorkerResult(content="Re-executed fast", result_type="code")
            await asyncio.sleep(0.01)
            return WorkerResult(content=f"Done: {task}", result_type="code")

        with patch.object(Worker, "analyze", new_callable=AsyncMock, side_effect=analyze_side_effect), \
             patch.object(Worker, "execute", new_callable=AsyncMock, side_effect=execute_side_effect), \
             patch.object(Worker, "synthesize", new_callable=AsyncMock) as mock_synth:
            mock_synth.return_value = WorkerResult(content="Combined")

            result = await tree.run("Task")

        straggler_events = [e for e in result.events if "STRAGGLER" in e]
        assert len(straggler_events) >= 1
        assert result.total_nodes >= 3

    @pytest.mark.asyncio
    async def test_no_straggler_when_all_fast(self, config):
        """No straggler detection when all children finish quickly."""
        config.straggler_multiplier = 2.0
        config.straggler_min_siblings = 2
        tree = TaskTree(config)

        call_count = {"analyze": 0}

        async def analyze_side_effect(node_id, task, context=""):
            call_count["analyze"] += 1
            if call_count["analyze"] == 1:
                return [
                    SubtaskRequest(task="A", reason="independent"),
                    SubtaskRequest(task="B", reason="independent"),
                    SubtaskRequest(task="C", reason="independent"),
                ]
            return []

        async def execute_side_effect(node_id, task, context="", approach=""):
            await asyncio.sleep(0.01)
            return WorkerResult(content=f"Done: {task}", result_type="code")

        with patch.object(Worker, "analyze", new_callable=AsyncMock, side_effect=analyze_side_effect), \
             patch.object(Worker, "execute", new_callable=AsyncMock, side_effect=execute_side_effect), \
             patch.object(Worker, "synthesize", new_callable=AsyncMock) as mock_synth:
            mock_synth.return_value = WorkerResult(content="Combined")

            result = await tree.run("Task")

        straggler_events = [e for e in result.events if "STRAGGLER" in e]
        assert len(straggler_events) == 0


class TestTokenBudget:
    """Tests for token budget enforcement."""

    @pytest.mark.asyncio
    async def test_budget_exhaustion_stops_execution(self, config):
        """When token budget is exceeded, execution stops immediately."""
        config.max_total_tokens = 100
        tree = TaskTree(config)

        # Pre-exhaust the budget
        tree.llm.cost_report.total_input_tokens = 100

        result = await tree.run("Some task")

        assert "budget exhausted" in result.answer.lower()
