"""Worker — two-phase execution unit with recursive splitting.

Each Worker runs in two phases:

  Phase 1 (ANALYZE): "Does this task have independent sub-problems?"
    → Yes: return spawn requests (the framework runs children, then calls Phase 2b)
    → No:  fall through to Phase 2a

  Phase 2a (EXECUTE): Do the actual work, return result.

  Phase 2b (SYNTHESIZE): Combine children's results into one coherent output.

This separation means:
- Analysis and execution don't interfere with each other
- Every node can still recurse (children also run Phase 1)
- Simple tasks skip Phase 1 quickly — one lightweight call
"""

from __future__ import annotations

import json
import logging

from .config import SproutConfig
from .llm import LLMLayer
from .types import Message, SubtaskRequest, WorkerResult

logger = logging.getLogger(__name__)

# ── Phase 1: Analysis ──────────────────────────────────────────────

ANALYZE_TOOL = {
    "type": "function",
    "function": {
        "name": "execution_plan",
        "description": (
            "Decide how to execute this task. Either handle it yourself, "
            "or split it into independent subtasks that can run in parallel."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "should_split": {
                    "type": "boolean",
                    "description": (
                        "true ONLY if the task contains 2+ genuinely independent "
                        "sub-problems that have NO dependencies between each other "
                        "and would benefit from parallel execution."
                    ),
                },
                "subtasks": {
                    "type": "array",
                    "description": "Independent subtasks (only when should_split=true)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Clear, self-contained description of WHAT to do",
                            },
                            "context": {
                                "type": "string",
                                "description": "All info this subtask needs (code, constraints, etc.)",
                            },
                            "approach": {
                                "type": "string",
                                "description": (
                                    "HOW the child should work: methodology, output format, "
                                    "what to focus on, what to avoid. This is like a brief "
                                    "for a team member — tell them the approach, not just the goal."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why this is independent from other subtasks",
                            },
                        },
                        "required": ["task", "reason"],
                    },
                },
            },
            "required": ["should_split"],
        },
    },
}

ANALYZE_SYSTEM_PROMPT = """You are analyzing a task to decide whether it should be split.

Your ONLY job right now is to call execution_plan. Do NOT do any actual work.

Split ONLY when ALL of these are true:
1. There are 2+ sub-problems that are genuinely INDEPENDENT
2. Sub-problem A's result does NOT affect sub-problem B's approach
3. Each sub-problem is complex enough to justify a separate worker
4. You can clearly describe each sub-problem with all necessary context

Do NOT split when:
- The task is a single coherent problem (even if it has multiple steps)
- Steps depend on each other (step 2 needs step 1's output)
- The task is simple enough to handle in one shot

When splitting:
- INCLUDE all relevant context (code snippets, constraints, data) in each subtask — the child worker won't see the parent's context.
- INCLUDE an "approach" for each subtask — tell the child worker HOW to work: what methodology to use, what output format, what to focus on, what pitfalls to avoid. Think of it like briefing a team member.

Call execution_plan now."""

# ── Phase 2a: Execute ──────────────────────────────────────────────

EXECUTE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "report_result",
            "description": "Report your final result and exit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Your complete output",
                    },
                    "result_type": {
                        "type": "string",
                        "enum": ["code", "analysis", "fix", "test", "plan", "general"],
                        "description": "Type of output",
                    },
                },
                "required": ["result"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abandon",
            "description": "Abandon this task — it's not useful or not feasible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you're abandoning",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]

EXECUTE_SYSTEM_PROMPT = """You are a focused worker. Do the work and report the result.

Rules:
- Do the actual work. Don't describe what you would do — DO it.
- Call report_result with your COMPLETE output when done.
- If not feasible, call abandon with the reason.
- No meta-commentary. Just output."""

# ── Phase 2b: Synthesize ──────────────────────────────────────────

SYNTHESIZE_SYSTEM_PROMPT = """You are synthesizing results from multiple subtasks into one coherent output.

Rules:
- Combine all subtask results into a single, well-organized response.
- Don't just list them — integrate them into a coherent whole.
- Resolve any contradictions (prefer the more detailed/correct one).
- Call report_result with the combined output."""


class Worker:
    """Two-phase execution unit. Stateless — create one per node."""

    def __init__(self, config: SproutConfig, llm: LLMLayer):
        self.config = config
        self.llm = llm

    async def analyze(
        self, node_id: str, task: str, context: str = "",
    ) -> list[SubtaskRequest]:
        """Phase 1: Analyze task and decide whether to split.

        Returns:
            [] → don't split, execute directly
            [SubtaskRequest, ...] → split into these parallel subtasks
        """
        user_content = f"## Task\n{task}"
        if context:
            user_content += f"\n\n## Context\n{context}"

        messages = [
            Message(role="system", content=ANALYZE_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        response = await self.llm.chat(
            messages=messages,
            agent_id=f"{node_id}",
            max_tokens=1024,
            tools=[ANALYZE_TOOL],
        )

        tool_calls = response.get("tool_calls")
        if not tool_calls:
            logger.info(f"Worker {node_id} analyze: no tool call → no split")
            return []

        for tc in tool_calls:
            if tc["function"]["name"] == "execution_plan":
                args = json.loads(tc["function"]["arguments"])

                if not args.get("should_split", False):
                    logger.info(f"Worker {node_id} analyze: no split")
                    return []

                subtasks = args.get("subtasks", [])
                if len(subtasks) < 2:
                    logger.info(f"Worker {node_id} analyze: <2 subtasks → no split")
                    return []

                requests = [
                    SubtaskRequest(
                        task=st["task"],
                        context=st.get("context", ""),
                        reason=st.get("reason", "independent subtask"),
                        approach=st.get("approach", ""),
                    )
                    for st in subtasks
                ]
                logger.info(f"Worker {node_id} analyze: split into {len(requests)} subtasks")
                return requests

        return []

    async def execute(
        self, node_id: str, task: str, context: str = "", approach: str = "",
    ) -> WorkerResult | None:
        """Phase 2a: Do the actual work.

        Returns:
            WorkerResult if completed, None if abandoned.
        """
        # Build system prompt — inject approach if parent provided one
        if approach:
            system = f"{EXECUTE_SYSTEM_PROMPT}\n\n## Your Approach\n{approach}"
        else:
            system = EXECUTE_SYSTEM_PROMPT

        user_content = f"## Task\n{task}"
        if context:
            user_content += f"\n\n## Context\n{context}"

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user_content),
        ]

        for _ in range(3):
            response = await self.llm.chat(
                messages=messages,
                agent_id=node_id,
                max_tokens=self.config.max_tokens_per_call,
                tools=EXECUTE_TOOLS,
            )

            content = response["content"]
            tool_calls = response["tool_calls"]

            if not tool_calls:
                if content.strip():
                    logger.info(f"Worker {node_id} execute: done (direct)")
                    return WorkerResult(content=content)
                messages.append(Message(role="assistant", content=content))
                messages.append(Message(role="user", content="Call report_result with your output."))
                continue

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])

                if fn_name == "report_result":
                    result = WorkerResult(
                        content=args["result"],
                        result_type=args.get("result_type", "general"),
                    )
                    logger.info(f"Worker {node_id} execute: done ({result.result_type})")
                    return result

                elif fn_name == "abandon":
                    logger.info(f"Worker {node_id} execute: abandoned — {args['reason']}")
                    return None

        logger.warning(f"Worker {node_id} execute: max iterations")
        return WorkerResult(content=content) if content.strip() else None

    async def synthesize(
        self, node_id: str, task: str, children_results: list[WorkerResult],
    ) -> WorkerResult | None:
        """Phase 2b: Synthesize children's results into one output.

        Only called when the node split into subtasks and needs to combine results.
        For simple cases (all children returned cleanly), considers skipping LLM.
        """
        # Optimization: if there's only structured results, just concatenate
        # without burning an LLM call
        if all(r.result_type in ("code", "fix") for r in children_results):
            combined = "\n\n---\n\n".join(r.content for r in children_results)
            logger.info(f"Worker {node_id} synthesize: direct concat (all code/fix)")
            return WorkerResult(content=combined, result_type="code")

        # Otherwise, use LLM to synthesize
        results_text = "\n\n---\n\n".join(
            f"### Subtask {i+1} ({r.result_type}):\n{r.content}"
            for i, r in enumerate(children_results)
        )
        user_content = (
            f"## Original Task\n{task}\n\n"
            f"## Subtask Results\n{results_text}\n\n"
            f"Synthesize these into one coherent output."
        )

        messages = [
            Message(role="system", content=SYNTHESIZE_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        response = await self.llm.chat(
            messages=messages,
            agent_id=node_id,
            max_tokens=self.config.max_tokens_per_call,
            tools=EXECUTE_TOOLS,
        )

        tool_calls = response.get("tool_calls")
        content = response["content"]

        if tool_calls:
            for tc in tool_calls:
                if tc["function"]["name"] == "report_result":
                    args = json.loads(tc["function"]["arguments"])
                    logger.info(f"Worker {node_id} synthesize: done")
                    return WorkerResult(
                        content=args["result"],
                        result_type=args.get("result_type", "general"),
                    )

        if content.strip():
            logger.info(f"Worker {node_id} synthesize: done (direct)")
            return WorkerResult(content=content)

        return None
