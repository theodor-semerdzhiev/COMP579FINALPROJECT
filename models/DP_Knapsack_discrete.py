import numpy as np
import math
from typing import List, Tuple, Dict, Any

def solve_knapsack_dp(
    weights: List[float],
    values:  List[float],
    capacity: float,
    decimals: int = 0
) -> Tuple[float, List[int], float]:
    """
    Approximate **upper‑bound** solution for 0‑1 knapsack using
    classic DP on *discretised* (integer‑scaled) weights.

    Parameters
    ----------
    weights  : list[float]
    values   : list[float]
    capacity : float
    decimals : int   (number of decimal places kept; higher → tighter, slower)

    Returns (best_value, best_set, capacity) where
      * best_value ≥ true optimum (because weights are rounded **down**)
      * best_set   gives indices chosen by DP (may overweight in originals!)
      * capacity   echoes the input
    """
    if capacity <= 0 or not weights:
        return 0.0, [], capacity

    # scaling factor (e.g. 10**2 keeps two decimal places)
    scale = 10 ** decimals

    w_int = [int(math.ceil((w * scale))) for w in weights]          # floor -> lighter
    cap_int = int(capacity * scale)

    n = len(weights)
    # 1‑D DP over weight dimension (value maximisation)
    dp = [0.0] * (cap_int + 1)
    keep = [[False] * n for _ in range(cap_int + 1)]   # track choices

    for i in range(n):
        wt = w_int[i]
        val = values[i]
        for c in range(cap_int, wt - 1, -1):           # iterate backward
            new_val = dp[c - wt] + val
            if new_val > dp[c]:
                dp[c] = new_val
                # copy previous keep row, then mark item i
                keep[c] = keep[c - wt].copy()
                keep[c][i] = True

    # best value is at cap_int
    best_value = dp[cap_int]

    # reconstruct chosen items
    chosen = [i for i, taken in enumerate(keep[cap_int]) if taken]

    return best_value, chosen, capacity

def solve_KP_instances_with_DP(problem_instances:List[Dict[str, Any]]) -> Tuple[List[List[int]], List[float], List[float]]:
    optimal_values = [0] * len(problem_instances)
    optimal_weight = [0] * len(problem_instances)
    optimal_sols_items = [None] * len(problem_instances)

    for i, instance in enumerate(problem_instances):
        if i % 100 == 0:
            print("Solved instance", i, "of", len(problem_instances))
        opt_sol, opt_items, opt_weight = solve_knapsack_dp(instance["values"], instance["weights"], instance["capacity"])
        optimal_values[i] = opt_sol
        optimal_weight[i] = opt_weight
        optimal_sols_items[i] = opt_items
    
    
    return optimal_sols_items, optimal_values, optimal_weight