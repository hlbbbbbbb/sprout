"""Tests for the two-phase Worker execution unit."""

from unittest.mock import AsyncMock, patch

import pytest

from sprout.config import SproutConfig
from sprout.llm import LLMLayer
from sprout.types import WorkerResult
from sprout.worker import Worker


@pytest.fixture
def config():
    return SproutConfig(model="test-model")


@pytest.fixture
def llm(config):
    return LLMLayer(model=config.model)


def _make_response(content="", tool_calls=None):
    return {
        "content": content,
        "tool_calls": tool_calls,
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 0.001,
    }


class TestAnalyze:
    """Phase 1: analyze() — decide whether to split."""

    @pytest.mark.asyncio
    async def test_no_tool_call_means_no_split(self, config, llm):
        """If LLM returns no tool call, don't split."""
        worker = Worker(config, llm)

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(content="I'll handle it myself")
            result = await worker.analyze("node-1", "Simple task")

        assert result == []

    @pytest.mark.asyncio
    async def test_should_split_false_means_no_split(self, config, llm):
        """execution_plan with should_split=false → no split."""
        worker = Worker(config, llm)

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(
                tool_calls=[{
                    "id": "tc_1",
                    "function": {
                        "name": "execution_plan",
                        "arguments": '{"should_split": false}',
                    },
                }],
            )
            result = await worker.analyze("node-1", "Single problem")

        assert result == []

    @pytest.mark.asyncio
    async def test_split_with_subtasks(self, config, llm):
        """execution_plan with should_split=true and subtasks → returns SubtaskRequests."""
        worker = Worker(config, llm)

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(
                tool_calls=[{
                    "id": "tc_1",
                    "function": {
                        "name": "execution_plan",
                        "arguments": '{"should_split": true, "subtasks": [{"task": "Fix auth.py", "reason": "Independent module"}, {"task": "Fix cart.py", "reason": "Independent module"}]}',
                    },
                }],
            )
            result = await worker.analyze("node-1", "Fix all bugs")

        assert len(result) == 2
        assert result[0].task == "Fix auth.py"
        assert result[1].task == "Fix cart.py"

    @pytest.mark.asyncio
    async def test_single_subtask_means_no_split(self, config, llm):
        """Less than 2 subtasks → don't split (not worth it)."""
        worker = Worker(config, llm)

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(
                tool_calls=[{
                    "id": "tc_1",
                    "function": {
                        "name": "execution_plan",
                        "arguments": '{"should_split": true, "subtasks": [{"task": "Only one", "reason": "test"}]}',
                    },
                }],
            )
            result = await worker.analyze("node-1", "Task")

        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_includes_context(self, config, llm):
        """Context is included in the user message sent to LLM."""
        worker = Worker(config, llm)

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(
                tool_calls=[{
                    "id": "tc_1",
                    "function": {
                        "name": "execution_plan",
                        "arguments": '{"should_split": false}',
                    },
                }],
            )
            await worker.analyze("node-1", "Fix bugs", context="File has 3 functions")

            call_args = mock.call_args
            messages = call_args.kwargs.get("messages") or call_args[0][0]
            user_msg = [m for m in messages if m.role == "user"][0]
            assert "File has 3 functions" in user_msg.content


class TestExecute:
    """Phase 2a: execute() — do the actual work."""

    @pytest.mark.asyncio
    async def test_direct_content_becomes_result(self, config, llm):
        """If LLM returns content without tool calls, it becomes the result."""
        worker = Worker(config, llm)

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(content="The answer is 42")
            result = await worker.execute("node-1", "What is the answer?")

        assert result is not None
        assert result.content == "The answer is 42"

    @pytest.mark.asyncio
    async def test_report_result_tool(self, config, llm):
        """Worker calls report_result → returns structured result."""
        worker = Worker(config, llm)

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(
                tool_calls=[{
                    "id": "tc_1",
                    "function": {
                        "name": "report_result",
                        "arguments": '{"result": "Fixed the bug", "result_type": "fix"}',
                    },
                }],
            )
            result = await worker.execute("node-1", "Fix the bug")

        assert result is not None
        assert result.content == "Fixed the bug"
        assert result.result_type == "fix"

    @pytest.mark.asyncio
    async def test_abandon_tool(self, config, llm):
        """Worker calls abandon → returns None."""
        worker = Worker(config, llm)

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(
                tool_calls=[{
                    "id": "tc_1",
                    "function": {
                        "name": "abandon",
                        "arguments": '{"reason": "Not feasible"}',
                    },
                }],
            )
            result = await worker.execute("node-1", "Do something impossible")

        assert result is None

    @pytest.mark.asyncio
    async def test_no_spawn_tool_in_execute(self, config, llm):
        """Execute phase has no spawn tools — only report_result and abandon."""
        worker = Worker(config, llm)

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(content="Done")
            await worker.execute("node-1", "Task")

            call_kwargs = mock.call_args
            tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
            tool_names = [t["function"]["name"] for t in tools]
            assert "spawn_subtask" not in tool_names
            assert "execution_plan" not in tool_names
            assert "report_result" in tool_names
            assert "abandon" in tool_names


class TestSynthesize:
    """Phase 2b: synthesize() — combine children results."""

    @pytest.mark.asyncio
    async def test_code_results_direct_concat(self, config, llm):
        """All code/fix results → direct concat without LLM call."""
        worker = Worker(config, llm)
        children = [
            WorkerResult(content="def fix_auth(): pass", result_type="code"),
            WorkerResult(content="def fix_cart(): pass", result_type="fix"),
        ]

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            result = await worker.synthesize("node-1", "Fix bugs", children)
            mock.assert_not_called()  # No LLM call needed

        assert result is not None
        assert "fix_auth" in result.content
        assert "fix_cart" in result.content
        assert result.result_type == "code"

    @pytest.mark.asyncio
    async def test_mixed_results_use_llm(self, config, llm):
        """Mixed result types → uses LLM to synthesize."""
        worker = Worker(config, llm)
        children = [
            WorkerResult(content="Auth analysis: weak hashing", result_type="analysis"),
            WorkerResult(content="def fix(): pass", result_type="code"),
        ]

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(
                tool_calls=[{
                    "id": "tc_1",
                    "function": {
                        "name": "report_result",
                        "arguments": '{"result": "Combined: found weak hashing, here is fix"}',
                    },
                }],
            )
            result = await worker.synthesize("node-1", "Fix bugs", children)

        assert result is not None
        assert "Combined" in result.content

    @pytest.mark.asyncio
    async def test_synthesize_direct_content(self, config, llm):
        """LLM returns content without tool call → still works."""
        worker = Worker(config, llm)
        children = [
            WorkerResult(content="Result A", result_type="analysis"),
            WorkerResult(content="Result B", result_type="general"),
        ]

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(content="A and B combined")
            result = await worker.synthesize("node-1", "Combine", children)

        assert result is not None
        assert result.content == "A and B combined"
