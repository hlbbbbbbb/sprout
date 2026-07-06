"""Offline rubric demo: prove the benchmark's scoring is deterministic.

This runs WITHOUT any network access or API key. It feeds two fixed,
hand-written "answers" through the exact same `evaluate_answer` rubric
used by `examples/benchmark.py`, and prints their scores.

  - GOOD_ANSWER  satisfies every rubric check across all 4 modules  → ~100/100
  - BAD_ANSWER   only completes 1 module (the tokenizer)            → ~25/100

The point: the rubric is pure regex matching over the answer text, so the
same input always yields the same score. The randomness in a real run comes
from the LLM's output, never from the grader.

Usage:
    python examples/rubric_demo.py
"""

from benchmark import evaluate_answer

# ── Fixture 1: a "good" answer that completes all 4 modules ────────────
# Only needs to contain the signals the rubric looks for — not a runnable
# program. Comments below map each block to its module.

GOOD_ANSWER = '''
# ===== Module 1: tokenizer.py =====
from dataclasses import dataclass


@dataclass
class Token:
    type: str
    value: str


def tokenize(expr: str) -> list[Token]:
    """Tokenize a math expression into NUMBER, IDENT, OP, LPAREN,
    RPAREN, COMMA tokens. A leading minus is unary/negative only when
    the previous token is not a NUMBER/IDENT/RPAREN."""
    tokens: list[Token] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        # ... dispatch to NUMBER / IDENT / OP / LPAREN / RPAREN / COMMA ...
        raise ValueError(f"unrecognized character: {ch!r}")
    return tokens


# ===== Module 2: lru_cache.py =====
import threading
import time
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int, default_ttl=None):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key][0]

    def put(self, key, value, ttl=None):
        with self._lock:
            expires = time.time() + (ttl or self.default_ttl or 1e18)
            self._data[key] = (value, expires)
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)  # evict least-recently-used

    def delete(self, key) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def clear(self):
        with self._lock:
            self._data.clear()

    def keys(self) -> list:
        with self._lock:
            return list(self._data.keys())

    def __len__(self):
        with self._lock:
            return len(self._data)


# ===== Module 3: csv_transformer.py =====
import csv
import io


def transform(csv_text: str, ops: list[dict]) -> str:
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    for op in ops:
        kind = op["op"]
        if kind == "filter":
            pass  # supports >, <, >=, <=, ==, !=, contains, startswith, endswith
        elif kind == "rename":
            pass
        elif kind == "add_column":
            template = op["expr"]
            template.format(**rows[0])  # {column} interpolation
        elif kind == "sort":
            pass
        elif kind == "drop":
            pass
        else:
            raise ValueError(f"unknown op: {kind}")
    return csv_text


# ===== Module 4: retry.py =====
import asyncio
import functools
import random


def retry(max_attempts=3, base_delay=1.0, max_delay=60.0,
          backoff_factor=2.0, exceptions=(Exception,), on_retry=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    delay = min(base_delay * backoff_factor ** attempt, max_delay)
                    delay *= random.uniform(0.5, 1.5)  # jitter
                    if on_retry:
                        on_retry(attempt, exc, delay)

        if asyncio.iscoroutinefunction(func):
            return wrapper
        return wrapper

    return decorator
'''

# ── Fixture 2: a "bad" answer that only completes 1 of 4 modules ───────
# Just the tokenizer; the other three modules are missing entirely.

BAD_ANSWER = '''
# Only got through the first module before running out of budget.

from dataclasses import dataclass


@dataclass
class Token:
    type: str
    value: str


def tokenize(expr: str) -> list[Token]:
    """Tokenize into NUMBER, IDENT, OP, LPAREN, RPAREN, COMMA.
    A leading minus is treated as a negative/unary sign when the
    previous token is not a value."""
    tokens: list[Token] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        raise ValueError("unrecognized character")
    return tokens

# TODO: lru_cache.py, csv_transformer.py, retry.py — not done.
'''


def _print_scores(label: str, answer: str) -> float:
    scores = evaluate_answer(answer)
    total = sum(scores.values())
    print(f"\n  {label}  (total {total:.1f} / 100)")
    print("  " + "-" * 42)
    for module, score in scores.items():
        print(f"    {module:<20} {score:>6.1f} / 25")
    return total


def main():
    print("=" * 50)
    print("  RUBRIC DEMO (offline, no API key, no network)")
    print("  Same evaluate_answer() as examples/benchmark.py")
    print("=" * 50)

    good_total = _print_scores("GOOD answer (all 4 modules)", GOOD_ANSWER)
    bad_total = _print_scores("BAD answer (only tokenizer)", BAD_ANSWER)

    print("\n" + "=" * 50)
    print(f"  Score gap: {good_total:.1f} vs {bad_total:.1f}"
          f"  (delta {good_total - bad_total:+.1f})")
    print("  The rubric is deterministic regex matching: the same")
    print("  answer text always produces the same score.")
    print("=" * 50)


if __name__ == "__main__":
    main()
