# 🌱 Sprout

> **Self-growing task tree for LLM agents. Each agent decides its own split — recursive, not predefined.**
>
> **LLM Agent 的自生长任务树。每个 Agent 自己决定分裂——递归的，非预设的。**

[English](#english) · [中文](#中文)

---

## English

### What is Sprout?

Sprout is a multi-agent framework where **agents grow their own task tree**. You give it one task, and each agent decides — mid-work — whether to split into smaller subtasks or finish the job itself. Children can split further, forming a tree as deep as the problem demands.

**Done tasks die. Results bubble up.** No predefined roles, no fixed topology.

### How it differs

| | CrewAI / AutoGen | DeerFlow (ByteDance) | Claude Code | **Sprout** |
|---|---|---|---|---|
| Topology | Manually wired graph | Star, fixed 2 layers | Fan-out, fixed 2 layers | **Tree, arbitrary depth** |
| Who decides split | Human designer | Lead Agent | Main Agent | **Every agent, autonomously** |
| Recursive splitting | ❌ | ❌ | ❌ | **✅** |
| Agent roles | Predefined | Predefined | N/A | **Emergent, injected at spawn time** |

### Core flow

```
User task
    │
    ▼
  Root node
    │
    ├─ Phase 1: analyze()        ← lightweight LLM call: should I split?
    │
    ├─ No split → Phase 2a: execute()        ← do the actual work
    │
    └─ Split → spawn children in parallel (each child runs the same flow!)
              │
              ├─ Child A → analyze → execute → result
              ├─ Child B → analyze → execute → result
              └─ (if a child is too slow → straggler detection → cancel → re-split)
              │
              └─ Phase 2b: synthesize()      ← merge children's results
```

Every node — root, child, grandchild — runs the same three-phase flow. That's what gives Sprout **true recursive splitting**, not a flat orchestrator → workers structure.

### Key mechanisms

- **Two-phase Worker** — `analyze()` decides whether to split (in a separate lightweight LLM call), `execute()` does the real work. This separation is what makes the split decision actually good.
- **Approach injection** — When a parent splits, it generates an `approach` field (methodology, output format, what to focus on) for each child. Children's system prompts are augmented with this approach. Roles emerge from the task, not from predefined prompts.
- **Straggler detection** — Replaces `asyncio.gather` with `asyncio.wait` + periodic checks. When one child takes more than 2.5× the average of its siblings, it gets cancelled and either re-split into finer subtasks or re-executed. Like a human team reassigning work when someone gets stuck.
- **Cheap synthesis** — If all child results are `code` or `fix` type, the parent skips the synthesize LLM call and just concatenates.
- **Safety limits** — `max_depth`, `max_children_per_node`, `max_total_nodes`, `max_total_tokens` prevent runaway trees.

### Quick start

```bash
git clone https://github.com/YangHuang2280/Sprout.git
cd Sprout
pip install -e .
```

```python
import asyncio
from sprout import TaskTree, SproutConfig

config = SproutConfig(
    model="openai/glm-4.5",             # any litellm-compatible model
    api_base="https://open.bigmodel.cn/api/paas/v4",
    max_depth=4,
    max_children_per_node=3,
)

async def main():
    result = await TaskTree(config).run(
        "Implement a Python CLI that converts CSV to JSON, with tests"
    )
    print(result.answer)
    print(result.tree_summary())   # visual tree of what happened

asyncio.run(main())
```

### When to use Sprout

✅ **Good fit**
- Tasks with clearly separable subtasks (multi-module codegen, multi-section reports, fan-out research)
- Output that exceeds a single LLM call's token budget
- Problems where the right decomposition isn't obvious upfront (let the agent figure it out)

❌ **Bad fit**
- Simple single-shot tasks (just call the LLM directly)
- Strictly sequential workflows (use LangGraph)
- Tasks requiring specific predesigned roles with specialized tools (use CrewAI / DeerFlow)

### Benchmark findings

We ran a controlled benchmark (`examples/benchmark.py`) comparing single-agent vs Sprout on a "write 4 independent Python modules" task:

- **When `max_tokens_per_call` is tight**: single agent can't finish all modules (25/100 quality score). Sprout splits and each child has its own token budget → 100/100.
- **When tokens are abundant**: quality is comparable; the speedup from parallelism is real but modest (network, rate limits).

**Takeaway: Sprout's main value isn't parallel speed — it's breaking past the token/attention bottleneck of a single LLM call.**

### Project layout

```
sprout/
├── sprout/
│   ├── __init__.py       # public API: TaskTree, SproutConfig, TreeResult
│   ├── types.py          # TaskNode, SubtaskRequest, WorkerResult, TreeResult
│   ├── config.py         # SproutConfig
│   ├── worker.py         # Two-phase Worker (analyze / execute / synthesize)
│   ├── task_tree.py      # Execution engine + straggler detection
│   └── llm.py            # litellm wrapper + concurrency semaphore
├── tests/                # 24 tests covering both units
└── examples/
    ├── code_fix.py       # multi-module bug fix demo
    └── benchmark.py      # single-agent vs Sprout comparison
```

### Tested models

- `openai/glm-4.5` (Zhipu, via litellm's OpenAI-compatible endpoint) — **recommended**, 10 concurrent
- `openai/glm-5.1` — works but only 1 concurrent
- Anything litellm supports (Claude, GPT-4o, DeepSeek, etc.) — one config line to switch

### License

MIT. See [LICENSE](LICENSE).

---

## 中文

### Sprout 是什么？

Sprout 是一个多 Agent 框架，让 **Agent 自己长出任务树**。给它一个任务，每个 Agent 在工作过程中自己决定——要不要分裂成更小的子任务，还是直接做掉。子 Agent 也可以继续分裂，树的深度由任务复杂度决定。

**做完就死，结果向上汇聚。** 没有预设角色，没有固定拓扑。

### 跟其他框架有啥区别

| | CrewAI / AutoGen | DeerFlow（字节） | Claude Code | **Sprout** |
|---|---|---|---|---|
| 拓扑 | 手工编排的图 | 星形，固定两层 | 扇形，固定两层 | **树形，深度不限** |
| 谁决定分裂 | 人类设计者 | Lead Agent | 主 Agent | **每个 Agent 自己** |
| 能否递归分裂 | ❌ | ❌ | ❌ | **✅** |
| Agent 角色 | 预设 | 预设 | 无 | **分裂时由父节点动态注入** |

### 核心执行流程

```
用户任务
    │
    ▼
  Root 节点
    │
    ├─ Phase 1: analyze()        ← 轻量 LLM 调用：要不要拆？
    │
    ├─ 不拆 → Phase 2a: execute()        ← 做实际工作
    │
    └─ 要拆 → 并行创建子节点（每个子节点走同样的流程！）
              │
              ├─ 子节点 A → analyze → execute → 结果
              ├─ 子节点 B → analyze → execute → 结果
              └─（某子节点太慢 → straggler 检测 → 取消 → 重新拆分）
              │
              └─ Phase 2b: synthesize()      ← 综合子结果
```

每个节点（包括子节点、孙节点）都走同样的三阶段流程——这是 Sprout **真正递归**的关键。不是"编排器 → 工人"的扁平结构。

### 核心机制

- **两阶段 Worker**：`analyze()` 只决定要不要拆（独立的轻量 LLM 调用），`execute()` 做实际工作。分析和执行分离，让分裂决策质量更高。
- **Approach 注入**：父节点分裂时，给每个子任务生成 `approach` 字段（方法论、输出格式、关注点）。子节点的 system prompt 会被注入这个 approach——角色从任务中涌现，不是预设的。
- **Straggler 检测**：用 `asyncio.wait` + 周期性检查替代 `asyncio.gather`。当某子任务耗时超过兄弟平均的 2.5 倍，取消并再拆分（或重新执行）。就像人类团队给卡住的成员加人帮忙。
- **综合优化**：如果子结果全是 `code` 或 `fix` 类型，父节点跳过综合阶段的 LLM 调用，直接拼接。
- **安全边界**：`max_depth`、`max_children_per_node`、`max_total_nodes`、`max_total_tokens` 防止树爆炸。

### 快速开始

```bash
git clone https://github.com/YangHuang2280/Sprout.git
cd Sprout
pip install -e .
```

```python
import asyncio
from sprout import TaskTree, SproutConfig

config = SproutConfig(
    model="openai/glm-4.5",             # 任何 litellm 兼容的模型
    api_base="https://open.bigmodel.cn/api/paas/v4",
    max_depth=4,
    max_children_per_node=3,
)

async def main():
    result = await TaskTree(config).run(
        "实现一个 Python CLI，把 CSV 转成 JSON，带测试"
    )
    print(result.answer)
    print(result.tree_summary())   # 可视化任务树

asyncio.run(main())
```

### 什么时候用 Sprout

✅ **适合的场景**
- 任务有明显可独立拆分的子部分（多模块代码生成、多章节报告、扇出式研究）
- 输出量超过单次 LLM 调用的 token 上限
- 分解方式不明显，想让 Agent 自己判断

❌ **不适合的场景**
- 简单的单次调用任务（直接调 LLM 就够了）
- 严格串行的工作流（用 LangGraph）
- 需要特定预设角色 + 专用工具的任务（用 CrewAI / DeerFlow）

### Benchmark 核心发现

跑了个对照实验（`examples/benchmark.py`），任务是"写 4 个独立的 Python 模块"，对比单 Agent vs Sprout：

- **当 `max_tokens_per_call` 紧张时**：单 Agent 写不完（质量分 25/100）。Sprout 分裂后每个子节点有独立 token 预算 → 100/100。
- **当 token 充足时**：质量基本持平，并行加速有但不显著（网络延迟 + 限流拖累）。

**结论：Sprout 的核心价值不是并行加速，而是突破单次 LLM 调用的 token / 注意力瓶颈。**

### 项目结构

```
sprout/
├── sprout/
│   ├── __init__.py       # 公开接口：TaskTree, SproutConfig, TreeResult
│   ├── types.py          # TaskNode, SubtaskRequest, WorkerResult, TreeResult
│   ├── config.py         # SproutConfig
│   ├── worker.py         # 两阶段 Worker（analyze / execute / synthesize）
│   ├── task_tree.py      # 执行引擎 + straggler 检测
│   └── llm.py            # litellm 封装 + 并发信号量
├── tests/                # 24 个测试
└── examples/
    ├── code_fix.py       # 多模块 bug 修复示例
    └── benchmark.py      # 单 Agent vs Sprout 对比
```

### 验证过的模型

- `openai/glm-4.5`（智谱，通过 litellm 的 OpenAI 兼容接口）——**推荐**，10 并发
- `openai/glm-5.1` —— 能用，但只 1 并发
- 任何 litellm 支持的模型（Claude、GPT-4o、DeepSeek 等）—— 改一行配置即可

### 许可证

MIT。详见 [LICENSE](LICENSE)。

---

*Made with 🌱 — agents that grow their own way.*
