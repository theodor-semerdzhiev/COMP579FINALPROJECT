from __future__ import annotations
"""Quick integration test for `knapsack_instance_generator` + `KnapsackEnv`.

Run this file directly:

    python test_knapsack_env.py

It will:
1. Create a small batch of instances from each family (RI, FI, HI, SS).
2. Feed each instance into the user's `KnapsackEnv` (imported from `your_env_module`).
3. Execute one random episode per instance and print a short summary.

Adjust `your_env_module` import below to match the actual path/name where
`KnapsackEnv` is defined (e.g. `from knapsack_env import KnapsackEnv`).
"""

from typing import Literal

import numpy as np

from util.instance_gen import KnapsackInstanceGenerator

# ---------------------------------------------------------------------------
# TODO: update this import to point at the file that defines KnapsackEnv.
# ---------------------------------------------------------------------------
from knapsackgym import KnapsackEnv  # noqa: F401 – replace as needed


# ---------------------------------------------------------------------------
# Helper: run one random episode and return the total reward
# ---------------------------------------------------------------------------

def run_random_episode(env: KnapsackEnv, max_steps: int | None = None) -> float:
    state, total_reward, done, steps = env.reset(), 0.0, False, 0
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return total_reward


# ---------------------------------------------------------------------------
# Main test driver
# ---------------------------------------------------------------------------

def main() -> None:
    gen = KnapsackInstanceGenerator(seed=2025)

    configs: dict[Literal["RI", "FI", "HI", "SS"], dict] = {
        "RI": dict(M=2, N=10, R=100),
        "FI": dict(M=2, N=10, capacity=2.5),  # tiny example
        "HI": dict(M=2, N=10, R=100),
        "SS": dict(M=2, N=10, R=50),
    }

    print("Running integration test (random policy)\n" + "-" * 40)

    for fam, kwargs in configs.items():
        instances = gen.generate(fam, **kwargs)
        print(f"Family {fam}: {len(instances)} instances")

        for j, inst in enumerate(instances, 1):
            # env.N must be >= number of items in instance
            env = KnapsackEnv(inst.to_dict(), N=len(inst.values))
            total_reward = run_random_episode(env)
            print(
                f"  Instance {j}: items={len(inst.values):2d}  "
                f"best_value={env.best_value:8.2f}  total_reward={total_reward:8.3f}"
            )

    print("\nAll tests completed – no exceptions raised.")


if __name__ == "__main__":
    main()
