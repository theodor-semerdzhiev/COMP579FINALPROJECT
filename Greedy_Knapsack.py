
def greedy_knapsack(problem_instance: dict) -> tuple:
    """
    Solve a 0-1 knapsack instance using a simple greedy strategy:
    1. Compute value-to-weight ratio for each item.
    2. Sort items by ratio in descending order.
    3. Pick items in that order as long as they fit in the knapsack.

    Args:
        problem_instance (dict): A dictionary with keys:
                                 - 'values': List of item values
                                 - 'weights': List of item weights
                                 - 'capacity': Max knapsack capacity

    Returns:
        tuple: (selected_items, total_value, total_weight)
               where selected_items is a list of item indices,
               total_value is the sum of the chosen items' values,
               and total_weight is the sum of the chosen items' weights.
    """

    values = problem_instance['values']
    weights = problem_instance['weights']
    capacity = problem_instance['capacity']

    # Build a list of (original_index, value, weight, ratio)
    # Avoid division by zero for items with weight == 0
    items = [
        (i, v, w, v / w if w > 0 else 0.0)
        for i, (v, w) in enumerate(zip(values, weights))
    ]

    # Sort by value-to-weight ratio in descending order
    items.sort(key=lambda x: x[3], reverse=True)

    selected_items = []
    remaining_capacity = capacity
    total_value = 0.0

    # Greedily pick items
    for idx, v, w, ratio in items:
        if w <= remaining_capacity:
            selected_items.append(idx)
            remaining_capacity -= w
            total_value += v

    total_weight = capacity - remaining_capacity

    return (selected_items, total_value, total_weight)

def solve_problem_instances_greedy(problem_instances: list) -> tuple:
    """
    Takes a list of knapsack problem instances and solves each of them
    with the greedy algorithm.

    Args:
        problem_instances (list of dict): Each dict has keys:
            - 'values': list of item values
            - 'weights': list of item weights
            - 'capacity': knapsack capacity

    Returns:
        tuple: (list_of_values, list_of_items_chosen, list_of_weights)
            - list_of_values: A list of total values (floats) for each instance
            - list_of_items_chosen: A list of lists, each containing the indices of chosen items
            - list_of_weights: A list of total weights (floats) for each instance
    """
    all_values = []
    all_chosen_items = []
    all_weights = []

    for instance in problem_instances:
        selected_items, total_value, total_weight = greedy_knapsack(instance)
        all_values.append(total_value)
        all_chosen_items.append(selected_items)
        all_weights.append(total_weight)

    return (all_values, all_chosen_items, all_weights)


# Example usage:
# if __name__ == "__main__":
#     problem_instances = [
#         {
#             'values': [10, 5, 15, 7, 6, 18, 3, 20],
#             'weights': [2, 3, 5, 7, 1, 4, 1, 5],
#             'capacity': 15
#         },
#         {
#             'values': [20, 30, 15, 25, 10, 14, 2, 6],
#             'weights': [5, 10, 8, 12, 4, 2, 5, 20],
#             'capacity': 20
#         }
#     ]

#     for i, instance in enumerate(problem_instances):
#         selected, val, wgt = greedy_knapsack(instance)
#         print(f"Greedy result for instance {i}:")
#         print(f"  Selected items: {selected}")
#         print(f"  Total value:   {val}")
#         print(f"  Total weight:  {wgt}\n")
