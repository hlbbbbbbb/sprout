"""Example: Self-growing task tree for code analysis and fixing.

The tree starts with one worker. If the worker discovers independent
sub-problems, it spawns children. Children may spawn their own children.
The tree grows organically based on task structure.

Usage:
    export ZHIPU_API_KEY="your-key"
    python examples/code_fix.py
"""

import asyncio
import logging
import os

from sprout import SproutConfig, TaskTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

TASK = """
## Buggy Code

The following two Python modules have bugs. They are independent of each other.

### Module 1: auth.py
```python
def validate_token(token, secret_key):
    import hashlib
    # Bug: should use hmac, not raw hash comparison (timing attack vulnerability)
    expected = hashlib.sha256(secret_key.encode()).hexdigest()
    return token == expected  # Bug: comparing wrong things

def login(username, password, db):
    user = db.get(username)
    if user is None:
        return {"error": "user not found"}
    # Bug: comparing plaintext password to hash
    if password == user["password_hash"]:
        return {"token": "generated_token"}
    return {"error": "wrong password"}
```

### Module 2: cart.py
```python
def calculate_total(items, discount_code=None):
    total = 0
    for item in items:
        # Bug: doesn't handle missing 'quantity' key
        total += item["price"] * item["quantity"]

    if discount_code:
        # Bug: discount applied as addition instead of subtraction
        total = total + total * 0.1

    # Bug: floating point — should round to 2 decimal places
    return total

def apply_tax(total, tax_rate=0.08):
    # Bug: tax_rate is applied wrong (should be total * (1 + rate))
    return total * tax_rate
```

## Task
1. Identify all bugs in both modules
2. Explain each bug
3. Write corrected versions
4. Add edge case handling
"""


async def main():
    config = SproutConfig(
        model="openai/glm-4.5",
        api_base="https://open.bigmodel.cn/api/paas/v4",
        api_key=os.environ.get("ZHIPU_API_KEY"),
        max_total_tokens=100_000,
        max_children_per_node=3,
        max_depth=3,
        max_total_nodes=8,
        verbose=True,
    )

    tree = TaskTree(config)
    result = await tree.run(TASK)

    # Print results
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result.answer)

    print("\n" + "-" * 60)
    print("TASK TREE")
    print("-" * 60)
    print(result.tree_summary)

    print(f"\nNodes: {result.total_nodes}")
    print(f"Max depth: {result.max_depth}")
    print(f"Spawns: {result.total_spawns}")
    print(f"Time: {result.elapsed_seconds:.1f}s")

    if result.cost_report:
        total_tokens = result.cost_report.total_input_tokens + result.cost_report.total_output_tokens
        print(f"Tokens: {total_tokens}")
        print(f"Cost: ${result.cost_report.total_cost_usd:.4f}")

    if result.events:
        print("\nLifecycle events:")
        for event in result.events:
            print(f"  {event}")


if __name__ == "__main__":
    asyncio.run(main())
