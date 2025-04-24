import numpy as np
from typing import List, Tuple, Dict, Any

def solve_knapsack_dp(
    weights: List[float],
    values:  List[float],
    capacity: float
) -> Tuple[float, List[int]]:
    """
    0-1 knapsack (exact) for real-valued weights/values — **inputs never mutated**.

    Parameters
    ----------
    weights  : list[float]   weight of each item
    values   : list[float]   value (profit) of each item
    capacity : float         maximum total weight

    Returns
    -------
    best_value : float       optimum total value
    chosen     : list[int]   indices (0-based) of items selected
    """
    # --- defensive copies -------------------------------------------------
    wts: List[float] = list(weights)   # shallow copy, enough for flat lists
    vals: List[float] = list(values)
    # ----------------------------------------------------------------------

    n = len(wts)
    if n == 0 or capacity <= 0:
        return 0.0, []

    # Sort items by value-to-weight ratio (needed for tight upper bounds)
    items = sorted(
        [(i, w, v, v / w) for i, (w, v) in enumerate(zip(wts, vals))],
        key=lambda t: t[3],
        reverse=True,
    )

    best_value = 0.0
    best_set: List[int] = []

    # Stack frames: (next_index, cur_value, cur_weight, picked_indices)
    stack: List[Tuple[int, float, float, List[int]]] = [(0, 0.0, 0.0, [])]

    while stack:
        idx, cur_val, cur_wt, picked = stack.pop()

        # If we've considered every item, check if this is a new best
        if idx == n:
            if cur_val > best_value:
                best_value, best_set = cur_val, picked
            continue

        # --- optimistic fractional-knapsack bound -------------------------
        bound_val, tmp_wt = cur_val, cur_wt
        j = idx
        while j < n and tmp_wt < capacity:
            _, w, v, _ = items[j]
            if tmp_wt + w <= capacity:
                bound_val += v
                tmp_wt += w
            else:
                bound_val += (capacity - tmp_wt) * (v / w)
                break
            j += 1

        # prune if even the optimistic bound can't beat best_value
        if bound_val <= best_value + 1e-12:
            continue

        # ------------------------------------------------------------------
        # Branch 1: take item `idx` (if it fits)
        orig_i, w_i, v_i, _ = items[idx]
        if cur_wt + w_i <= capacity:
            stack.append(
                (idx + 1, cur_val + v_i, cur_wt + w_i, picked + [orig_i])
            )

        # Branch 2: skip item `idx`
        stack.append((idx + 1, cur_val, cur_wt, picked))

    best_set.sort()          # ascending order for convenience
    return best_value, best_set, capacity

def solve_KP_instances_with_DP(problem_instances:List[Dict[str, Any]]) -> Tuple[List[List[int]], List[float], List[float]]:
    optimal_values = [0] * len(problem_instances)
    optimal_weight = [0] * len(problem_instances)
    optimal_sols_items = [None] * len(problem_instances)

    for i, instance in enumerate(problem_instances):
        opt_sol, opt_items, opt_weight = solve_knapsack_dp(instance["values"], instance["weights"], instance["capacity"])
        optimal_values[i] = opt_sol
        optimal_weight[i] = opt_weight
        optimal_sols_items[i] = opt_items
    
    
    return optimal_sols_items, optimal_values, optimal_weight