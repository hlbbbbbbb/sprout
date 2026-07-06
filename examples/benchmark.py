"""Benchmark: Single-Agent vs Multi-Agent on the same task.

Compares wall clock time, token usage, and output quality
between single-agent (no splitting) and multi-agent (tree splitting).

Usage:
    export ZHIPU_API_KEY="your-key"
    python examples/benchmark.py
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass

from sprout import SproutConfig, TaskTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── Benchmark Task ─────────────────────────────────────────────────

BENCHMARK_TASK = """
## Task: Implement a mini data-processing toolkit

Implement these 4 independent Python modules. Each must be a complete,
working module with proper error handling and edge cases.

### Module 1: `tokenizer.py`
A simple tokenizer for a subset of mathematical expressions.
- Function `tokenize(expr: str) -> list[Token]` where Token is a
  dataclass with fields (type: str, value: str).
- Token types: NUMBER (int or float, including negatives like -3.14),
  IDENT (variable names like x, foo), OP (+, -, *, /, ^, %),
  LPAREN, RPAREN, COMMA.
- Skip whitespace. Raise ValueError on unrecognized characters.
- Handle edge cases: empty string returns [], "---3" should tokenize
  as OP(-) OP(-) NUMBER(-3) — i.e., a leading minus before a digit
  is part of the number ONLY if not preceded by another number/ident/rparen.

### Module 2: `lru_cache.py`
An LRU cache with TTL (time-to-live) support.
- Class `LRUCache(capacity: int, default_ttl: float | None = None)`
- Methods: `get(key) -> value | None`, `put(key, value, ttl=None)`,
  `delete(key) -> bool`, `clear()`, `__len__()`, `keys() -> list`.
- Eviction: when capacity is exceeded, evict the least-recently-used
  item (get counts as "use"). Expired items are evicted first.
- Thread-safe using threading.Lock.
- `keys()` must not return expired keys.

### Module 3: `csv_transformer.py`
A CSV transformation pipeline.
- Function `transform(csv_text: str, ops: list[dict]) -> str`
- ops is a list of operations applied in order. Each op is a dict:
  - {"op": "filter", "column": "age", "predicate": ">30"} — keep rows
    where column matches predicate (support >, <, >=, <=, ==, !=,
    contains, startswith, endswith; numeric comparisons for numeric values)
  - {"op": "rename", "from": "old_name", "to": "new_name"}
  - {"op": "add_column", "name": "full_name", "expr": "{first} {last}"}
    — expr uses {column_name} interpolation
  - {"op": "sort", "column": "age", "order": "desc"}
  - {"op": "drop", "columns": ["col1", "col2"]}
- Return transformed CSV as string. Raise ValueError on unknown ops
  or missing columns.

### Module 4: `retry.py`
A retry decorator with exponential backoff.
- Decorator `@retry(max_attempts=3, base_delay=1.0, max_delay=60.0,
  backoff_factor=2.0, exceptions=(Exception,), on_retry=None)`
- Exponential backoff: delay = min(base_delay * backoff_factor^attempt, max_delay)
- Add jitter: multiply delay by random uniform(0.5, 1.5).
- on_retry is an optional callback: on_retry(attempt, exception, delay).
- Support both sync and async functions (detect with asyncio.iscoroutinefunction).
- After final failure, raise the last exception with __cause__ chain.
- The decorator must preserve the original function's signature (use functools.wraps).

## Requirements
- Each module must be fully self-contained (no cross-imports).
- Include the Token dataclass definition in tokenizer.py.
- Use only stdlib modules.
- Code must be production-quality: type hints, docstrings, edge case handling.
"""

# ── Quality Evaluator ──────────────────────────────────────────────


def score_tokenizer(text: str) -> float:
    score = 0.0
    if re.search(r"(dataclass|class\s+Token)", text):
        score += 5
    if re.search(r"def\s+tokenize\s*\(\s*expr\s*:", text):
        score += 5
    token_types = ["NUMBER", "IDENT", "OP", "LPAREN", "RPAREN", "COMMA"]
    found = sum(1 for t in token_types if t in text)
    score += (found / len(token_types)) * 5
    if re.search(r"(prev|previous|last).*token", text, re.IGNORECASE) or \
       re.search(r"unary|negative|leading.*(minus|dash)", text, re.IGNORECASE):
        score += 5
    if re.search(r"ValueError|raise\s+ValueError", text):
        score += 5
    return score


def score_lru_cache(text: str) -> float:
    score = 0.0
    if re.search(r"class\s+LRUCache", text):
        score += 5
    methods = ["def get", "def put", "def delete", "def clear", "def keys", "__len__"]
    found = sum(1 for m in methods if m in text)
    score += (found / len(methods)) * 5
    if re.search(r"ttl|time_to_live|expir", text, re.IGNORECASE):
        score += 2.5
    if re.search(r"time\.time\(\)|time\.monotonic\(\)|datetime", text):
        score += 2.5
    if re.search(r"threading\.(Lock|RLock)|Lock\(\)", text):
        score += 5
    if re.search(r"OrderedDict|move_to_end|_evict|pop.*least", text, re.IGNORECASE):
        score += 5
    return score


def score_csv_transformer(text: str) -> float:
    score = 0.0
    if re.search(r"def\s+transform\s*\(", text):
        score += 5
    ops = ["filter", "rename", "add_column", "sort", "drop"]
    found = sum(1 for op in ops if re.search(rf"""['\"]({op})['\"]""", text))
    score += (found / len(ops)) * 5
    predicates = [">=", "<=", "!=", "contains", "startswith", "endswith"]
    found = sum(1 for p in predicates if p in text)
    score += min((found / 4) * 5, 5)
    if re.search(r"import csv|csv\.(reader|writer|DictReader)", text):
        score += 5
    if re.search(r"format|\.format\(|f['\"]|str\.replace|\{.*\}", text):
        score += 5
    return score


def score_retry(text: str) -> float:
    score = 0.0
    if re.search(r"def\s+retry\s*\(", text):
        score += 5
    if re.search(r"backoff_factor\s*\*\*|base_delay\s*\*|pow\(|\*\*\s*attempt", text):
        score += 5
    if re.search(r"random\.(uniform|random)|jitter", text, re.IGNORECASE):
        score += 5
    if re.search(r"iscoroutinefunction|asyncio|async\s+def\s+wrapper", text):
        score += 5
    if re.search(r"functools\.wraps|@wraps", text):
        score += 5
    return score


def evaluate_answer(answer: str) -> dict[str, float]:
    return {
        "tokenizer": score_tokenizer(answer),
        "lru_cache": score_lru_cache(answer),
        "csv_transformer": score_csv_transformer(answer),
        "retry": score_retry(answer),
    }


# ── Runner ─────────────────────────────────────────────────────────

SHARED_CONFIG = dict(
    model="openai/glm-4.5",
    api_base="https://open.bigmodel.cn/api/paas/v4",
    api_key=os.environ.get("ZHIPU_API_KEY"),
    max_total_tokens=200_000,
    max_tokens_per_call=16384,
    verbose=True,
)


@dataclass
class BenchmarkResult:
    mode: str
    wall_clock_seconds: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    total_nodes: int
    max_depth: int
    tree_summary: str
    answer: str
    quality_scores: dict
    quality_total: float
    events: list


async def run_mode(mode: str) -> BenchmarkResult:
    if mode == "single":
        config = SproutConfig(
            **SHARED_CONFIG,
            max_depth=1,              # root cannot split
            max_children_per_node=0,
            max_total_nodes=1,
        )
    else:
        config = SproutConfig(
            **SHARED_CONFIG,
            max_depth=3,
            max_children_per_node=5,
            max_total_nodes=10,
        )

    tree = TaskTree(config)
    result = await tree.run(BENCHMARK_TASK)

    cost = result.cost_report
    total_tokens = cost.total_input_tokens + cost.total_output_tokens

    quality_scores = evaluate_answer(result.answer)

    return BenchmarkResult(
        mode=mode,
        wall_clock_seconds=result.elapsed_seconds,
        total_tokens=total_tokens,
        input_tokens=cost.total_input_tokens,
        output_tokens=cost.total_output_tokens,
        total_nodes=result.total_nodes,
        max_depth=result.max_depth,
        tree_summary=result.tree_summary,
        answer=result.answer,
        quality_scores=quality_scores,
        quality_total=sum(quality_scores.values()),
        events=result.events,
    )


def print_report(single: BenchmarkResult, multi: BenchmarkResult):
    speed_ratio = single.wall_clock_seconds / max(multi.wall_clock_seconds, 0.1)
    token_ratio = multi.total_tokens / max(single.total_tokens, 1)
    quality_delta = multi.quality_total - single.quality_total

    print("\n" + "=" * 65)
    print("  BENCHMARK RESULTS: Single-Agent vs Multi-Agent")
    print("=" * 65)
    print(f"  {'Metric':<28} {'Single':>15} {'Multi':>15}")
    print("  " + "-" * 58)
    print(f"  {'Wall clock (s)':<28} {single.wall_clock_seconds:>15.1f} {multi.wall_clock_seconds:>15.1f}")
    print(f"  {'Total tokens':<28} {single.total_tokens:>15,} {multi.total_tokens:>15,}")
    print(f"  {'  Input tokens':<28} {single.input_tokens:>15,} {multi.input_tokens:>15,}")
    print(f"  {'  Output tokens':<28} {single.output_tokens:>15,} {multi.output_tokens:>15,}")
    print(f"  {'Nodes':<28} {single.total_nodes:>15} {multi.total_nodes:>15}")
    print(f"  {'Max depth':<28} {single.max_depth:>15} {multi.max_depth:>15}")
    print(f"  {'Quality (total / 100)':<28} {single.quality_total:>15.1f} {multi.quality_total:>15.1f}")
    print("  " + "-" * 58)

    for module in ["tokenizer", "lru_cache", "csv_transformer", "retry"]:
        sq = single.quality_scores.get(module, 0)
        mq = multi.quality_scores.get(module, 0)
        delta = mq - sq
        indicator = "  " if delta == 0 else (" ↑" if delta > 0 else " ↓")
        print(f"    Quality: {module:<20} {sq:>11.1f} {mq:>11.1f}{indicator}")

    print("  " + "-" * 58)
    print(f"  {'Quality delta':<28} {quality_delta:>+15.1f}  (multi - single)")
    print(f"  {'Speed ratio':<28} {speed_ratio:>15.2f}x (>1 = multi faster)")
    print(f"  {'Token ratio':<28} {token_ratio:>15.2f}x (multi / single)")
    print("=" * 65)

    # Verdict
    print("\n  VERDICT:")
    if quality_delta > 5:
        print(f"  Quality: Multi-agent wins (+{quality_delta:.0f} points)")
    elif quality_delta < -5:
        print(f"  Quality: Single-agent wins ({quality_delta:.0f} points)")
    else:
        print(f"  Quality: Roughly equal (delta {quality_delta:+.0f})")

    if speed_ratio > 1.2:
        print(f"  Speed:   Multi-agent is {speed_ratio:.1f}x faster")
    elif speed_ratio < 0.8:
        print(f"  Speed:   Single-agent is {1/speed_ratio:.1f}x faster")
    else:
        print(f"  Speed:   Roughly equal ({speed_ratio:.2f}x)")

    if token_ratio > 1.3:
        print(f"  Cost:    Multi-agent uses {token_ratio:.1f}x more tokens")
    elif token_ratio < 0.7:
        print("  Cost:    Multi-agent uses fewer tokens")
    else:
        print(f"  Cost:    Roughly equal ({token_ratio:.2f}x)")

    # Tree structures
    print("\n" + "-" * 65)
    print("  SINGLE-AGENT TREE:")
    print("  " + single.tree_summary.replace("\n", "\n  "))
    print("\n  MULTI-AGENT TREE:")
    print("  " + multi.tree_summary.replace("\n", "\n  "))

    if multi.events:
        print("\n  MULTI-AGENT LIFECYCLE EVENTS:")
        for event in multi.events:
            print(f"    {event}")


async def main():
    print("=" * 65)
    print("  SPROUT BENCHMARK: Single-Agent vs Multi-Agent")
    print("  Task: Implement 4 independent Python modules")
    print(f"  Model: {SHARED_CONFIG['model']}")
    print("=" * 65)

    print("\n>>> Running SINGLE-AGENT mode (no splitting)...")
    single = await run_mode("single")
    print(f"    Done. {single.wall_clock_seconds:.1f}s, {single.total_tokens} tokens, quality={single.quality_total:.0f}/100")

    # Brief pause between runs to avoid rate limits
    print("\n>>> Waiting 5s before next run...")
    await asyncio.sleep(5)

    print("\n>>> Running MULTI-AGENT mode (splitting enabled)...")
    multi = await run_mode("multi")
    print(f"    Done. {multi.wall_clock_seconds:.1f}s, {multi.total_tokens} tokens, quality={multi.quality_total:.0f}/100")

    print_report(single, multi)


if __name__ == "__main__":
    asyncio.run(main())
