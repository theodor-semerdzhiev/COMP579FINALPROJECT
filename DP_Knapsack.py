import numpy as np
from typing import List, Tuple, Dict, Any

def solve_knapsack_dp(problem: Dict[str, Any]) -> Tuple[float, List[int], float]:
    """
    Solves the 0-1 Knapsack Problem using Dynamic Programming.
    
    Args:
        problem (Dict): A dictionary containing:
            - 'values': List of item values
            - 'weights': List of item weights
            - 'capacity': Knapsack capacity
    
    Returns:
        Tuple[float, List[int]]: A tuple containing:
            - The maximum value achievable
            - A list of indices of the selected items
    """
    values = problem['values']
    weights = problem['weights']
    capacity = problem['capacity']
    n = len(values)
    
    # Create DP table: rows are items, columns are capacities (0 to W)
    dp = np.zeros((n + 1, capacity + 1), dtype=float)
    
    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # If item i-1 can fit in knapsack of capacity w
            if weights[i-1] <= w:
                # Max of (including this item, excluding this item)
                dp[i, w] = max(values[i-1] + dp[i-1, w-weights[i-1]], dp[i-1, w])
            else:
                # Cannot include this item
                dp[i, w] = dp[i-1, w]
    
    # Backtrack to find the selected items
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        # If this item was included
        if dp[i, w] != dp[i-1, w]:
            selected_items.append(i-1)  # Item index is i-1
            w -= weights[i-1]
    
    # Reverse the list to get items in original order
    selected_items.reverse()
    
    total_weight = 0
    for item in selected_items:
        total_weight += weights[item]

    return float(dp[n, capacity]), selected_items, float(total_weight)


def solve_KP_instances_with_DP(problem_instances:List[Dict[str, Any]]) -> Tuple[List[List[int]], List[float], List[float]]:
    optimal_values = [0] * len(problem_instances)
    optimal_weight = [0] * len(problem_instances)
    optimal_sols_items = [None] * len(problem_instances)

    for i, instance in enumerate(problem_instances):
        opt_sol, opt_items, opt_weight = solve_knapsack_dp(instance)
        optimal_values[i] = opt_sol
        optimal_weight[i] = opt_weight
        optimal_sols_items[i] = opt_items
    
    
    return optimal_sols_items, optimal_values, optimal_weight
    
