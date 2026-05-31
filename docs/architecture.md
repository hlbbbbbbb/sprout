# Architecture

Sprout grows a **task tree**. You hand it one task; every node decides for itself
whether to split into independent subtasks or do the work directly. Children
recurse through the same logic, so the tree deepens only as far as the problem
demands. There is no orchestrator role and no predefined topology — the shape
comes from the task.

```
TaskTree.run(task)
        │
        ▼
   ┌───────────────────────────────────────────────┐
   │  _execute_node(node)   ← same for every node   │
   │                                                 │
   │  Phase 1   analyze()      should I split?       │
   │              │                                  │
   │     ┌────────┴────────┐                         │
   │     │ no              │ yes                      │
   │     ▼                 ▼                          │
   │  Phase 2a          spawn children               │
   │  execute()         (each recurses into          │
   │  do the work        _execute_node)              │
   │                       │                          │
   │                       ▼                          │
   │                    Phase 2b  synthesize()        │
   │                    merge child results           │
   └─────────────────────────────────────────────────┘
```

The execution engine is in [`sprout/task_tree.py`](../sprout/task_tree.py); the
per-node logic is in [`sprout/worker.py`](../sprout/worker.py).

## The node lifecycle

Every node — root, child, grandchild — runs the identical flow in
`TaskTree._execute_node()`:

1. **Budget gate.** If the cumulative token budget (`max_total_tokens`) is spent,
   the node short-circuits and returns immediately.
2. **`analyze()` — Phase 1.** A *lightweight* LLM call whose only job is to call
   the `execution_plan` tool and answer "does this task contain 2+ genuinely
   independent sub-problems?" It does **no actual work**. Returns either `[]`
   (don't split) or a list of `SubtaskRequest`s. A split is only attempted while
   the node is under `max_depth` and the tree is under `max_total_nodes`.
3. **`execute()` — Phase 2a.** If there was no split, the node does the real work
   via the `report_result` / `abandon` tools (a short tool loop, capped at
   `max_tokens_per_call`).
4. **Spawn → recurse → `synthesize()` — Phase 2b.** If it split, children are
   created and run **concurrently** (see straggler detection below); each child
   re-enters `_execute_node` and may split again. When children finish, the
   parent synthesizes their results into one answer.

The result bubbles up: a node's answer is either its own `execute()` output or the
`synthesize()` of its children. `TaskTree.run()` returns a `TreeResult` with the
final answer, a rendered tree, node/depth/spawn counts, a `CostReport`, and a
lifecycle event log.

## Why two phases

The single most important design decision. In v0.1 the "decide whether to split"
judgment and the "do the work" generation lived in the **same** LLM call. Models
handled this badly — weaker ones never split; stronger ones split poorly (e.g.
decomposing 1/3 of an obviously-separable task) because the two objectives compete
for the same context and attention.

Splitting them fixes it:

- **`analyze()`** sees a prompt that says, in effect, *"your only job is to decide;
  do not do any work."* Cheap call, sharp decision.
- **`execute()`** sees a prompt that says *"do the work, report the result."*

Crucially, analysis stays **inside the Worker**, not in a separate global Planner.
A standalone planner would collapse Sprout back into the flat two-layer
"planner → workers" shape of DeerFlow / Claude Code. Keeping `analyze()` per-node
is exactly what preserves **recursive** splitting.

## Approach injection — where roles come from

When a parent splits, `analyze()` produces an `approach` string for each child
(methodology, output format, what to focus on, what to avoid — like a brief to a
teammate). The child's system prompt in `execute()` is augmented with that
approach. So roles are **emergent and injected at spawn time**, not drawn from a
predefined cast of agents. The parent that understands the whole task shapes how
each child works.

## Straggler detection

Parallel children run with `asyncio.wait` + a periodic check loop
(`_execute_children_with_straggler_detection`), not `asyncio.gather`. `gather` can
only wait; it can't intervene when one child is dragging.

Every ~10s the loop:

1. collects newly finished children and records each one's elapsed time;
2. once at least `straggler_min_siblings` (default 2) have finished, computes the
   average sibling completion time;
3. flags any still-pending child whose elapsed time exceeds
   `avg × straggler_multiplier` (default 2.5×) as a **straggler**.

A flagged straggler is **cancelled**, then re-`analyze()`d:

- if it now splits into ≥2 subtasks → those grandchildren run in parallel and are
  synthesized into the straggler's result;
- if it doesn't → it is simply re-`execute()`d (the first attempt may have just
  stalled on a slow call).

This mirrors a human team noticing someone is stuck and either breaking their work
down further or reassigning it — instead of everyone idling on the slowest member.

## Cheap synthesis

`synthesize()` skips the LLM entirely when every child result is of type `code` or
`fix`: structured outputs are just concatenated. No reason to pay for a model call
to glue together code blocks. Other result types fall through to an LLM synthesis
pass.

## Safety limits

All splitting is bounded so the tree can't explode. Defaults live in
[`sprout/config.py`](../sprout/config.py):

| Setting | Default | Meaning |
|---|---|---|
| `max_depth` | 4 | deepest the tree may grow |
| `max_children_per_node` | 3 | most subtasks one node may spawn |
| `max_total_nodes` | 15 | hard cap on nodes in the whole tree |
| `max_total_tokens` | 500_000 | cumulative token budget (hard stop) |
| `max_tokens_per_call` | 4096 | max output tokens per LLM call |
| `max_retries` | 1 | retries on an LLM call failure |
| `straggler_multiplier` | 2.5 | "slower than `avg ×` this" → straggler |
| `straggler_min_siblings` | 2 | finished siblings required before straggler checks start |

Hitting any structural limit makes `analyze()` get skipped, so the node simply
executes and cannot split further. All LLM calls flow through `LLMLayer`
([`sprout/llm.py`](../sprout/llm.py)), a thin litellm wrapper that tracks
tokens/cost per node and bounds concurrency with an `asyncio` semaphore.

## Design decisions

| Decision | Choice | Why | Rejected |
|---|---|---|---|
| Agent communication | Tree (parent↔child) | Clear information flow, no global state | Shared blackboard (v0.1) |
| Who decides to spawn | The agent itself (tool call) | An agent only knows mid-work whether a task is separable | External rule engine (v0.1) |
| Agent state | Stateless `Worker` | Simple, reproducible, testable | Stateful agents with roles + history (v0.1) |
| Parallelism | `asyncio.wait` + straggler detection | Can intervene on slow children | `asyncio.gather` (can't intervene) |
| Model interface | litellm | One line to switch models | Per-vendor SDKs |
| Split decision | Two-phase Worker (analyze + execute) | Separates concerns *and* keeps recursion | Standalone Planner (flattens to 2 layers) |
| Synthesis | Concatenate `code`/`fix` directly | Saves tokens; structured output needs no rewrite | Always synthesize via LLM |

## How Sprout compares

| | CrewAI / AutoGen | DeerFlow | Claude Code | **Sprout** |
|---|---|---|---|---|
| Topology | Hand-wired graph | Star, fixed 2 layers | Fan-out, fixed 2 layers | **Tree, arbitrary depth** |
| Who decides the split | Human designer | Lead agent | Main agent | **Every agent** |
| Recursive splitting | ❌ | ❌ | ❌ | **✅** |
| Roles | Predefined | Predefined | N/A | **Emergent, injected at spawn** |

## Evolution: from swarm to tree

Sprout started (v0.1) as an ant-colony imitation: a shared blackboard, heat decay
on shared items, and an external `LifecycleManager` applying spawn/merge/kill
rules. It worked, but it was a **rule engine wearing an agent costume** — the
"autonomy" lived in human-written heuristics, not in the agents.

v0.2 threw that out for the task-tree model in this repo: no blackboard, no
external lifecycle rules, decisions made by each node's own `analyze()` call. The
first-principles analysis that drove the pivot — what multi-agent systems
fundamentally *are*, what actually bottlenecks them, and what is genuinely novel
versus repackaged — is written up in [`design-notes.md`](design-notes.md).

The headline finding from [`examples/benchmark.py`](../examples/benchmark.py):
Sprout's value is **not** parallel speed. It's breaking past the token/attention
ceiling of a single LLM call — when one call can't fit the whole job, splitting
lets each child spend its own budget on its own piece.
