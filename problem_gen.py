import random

# A simple function for now, but Bohan this is your job
def create_knapsack_problem_instances(
    num_instances: int, 
    N: int, 
    value_range: tuple = (1, 50), 
    weight_range: tuple = (1, 50),
    capacity_range: tuple = (50, 200),
    seed: int = None
) -> list:
    """
    Create a list of random knapsack problem instances, each containing:
        - A list of item values
        - A list of item weights
        - A single capacity value
        
    Each instance will have at most N items.

    Args:
        num_instances (int): Number of problem instances to create.
        N (int): Maximum number of items in each instance.
        value_range (tuple): Min and max possible item value (inclusive).
        weight_range (tuple): Min and max possible item weight (inclusive).
        capacity_range (tuple): Min and max knapsack capacity (inclusive).
        seed (int): Optional random seed for reproducibility. 
                    Defaults to None (no seeded behavior).

    Returns:
        List[dict]: A list of dictionaries, each with keys:
                    'values', 'weights', and 'capacity'.
    """
    if seed is not None:
        random.seed(seed)
    
    problem_instances = []
    for _ in range(num_instances):
        # Random number of items up to N
        num_items = random.randint(1, N)

        # Generate values and weights
        values = [random.randint(value_range[0], value_range[1]) for _ in range(num_items)]
        weights = [random.randint(weight_range[0], weight_range[1]) for _ in range(num_items)]
        
        # Generate capacity
        capacity = random.randint(capacity_range[0], capacity_range[1])
        
        problem_instance = {
            'values': values,
            'weights': weights,
            'capacity': capacity
        }
        problem_instances.append(problem_instance)

    return problem_instances
