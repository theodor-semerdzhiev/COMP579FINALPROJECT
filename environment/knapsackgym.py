import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, Tuple, List, Any, Callable


"""
Set of functions that determine the negative and positve reward functions, this is needed for certain experiments
Negative Reward: When selected object goes over weight limit
Positive Reward: When selected DOES NOT go over weight limit 
"""

def wr_i_negative_reward(value:float, weight:float, capacity:float) -> float:
    return (value / capacity)

def vr_i_positive_reward(value:float, weight:float, capacity:float) -> float:
    return value / (weight * capacity)

def w_i_negative_reward(value:float, weight:float, capacity:float) -> float:
    return weight

def v_i_positive_reward(value:float, weight:float, capacity:float) -> float:
    return value

def _1_positive_reward(value:float, weight:float, capacity:float) -> float:
    return 1.0
def _1_negative_reward(value:float, weight:float, capacity:float) -> float:
    return 1.0

class KnapsackEnv(gym.Env):
    """
    A Gym environment for the 0-1 Knapsack Problem.
    This version maintains a state vector of size (2*n + 4) for n items:
      - The first 2*n slots hold pairs of (value_i, weight_i) for each item, left-aligned.
      - The last 4 slots hold [capacity, sum(values), sum(weights), n_items].
    After an item is chosen (whether it fits or not), it is removed from the state by
    'shifting' all subsequent items left by 2, and placing zeros at the end.

    Reward is normalized:
      - v_r_i = v_i / (w_i * W_P) if the item fits.
      - -w_r_i = - (w_i / W_P) if the item does not fit.
      - -W_P if the action is out of range.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, problem_instance: dict[str, Any], N=100,
                 positive_reward_function:Callable[[float, float,float], float]=vr_i_positive_reward,
                 negative_reward_function:Callable[[float, float,float], float]=wr_i_negative_reward,
                 pad_meta_data_at_back:bool=True
                 ):
        """
        Initialize the Knapsack environment with a single problem instance.

        Args:
            problem_instance (Dict): Must have:
                - 'values': List[float] of item values
                - 'weights': List[float] of item weights
                - 'capacity': float (max knapsack capacity)
            
                N: Max number of items that a knapsack can have
                positive_reward_function: Reward function when we have a positive reward
                negative_reward_function: Reward function when we have a negative reward
                pad_meta_data_at_back: wehther we should put the weight, value sums, etc, at the back of state tensor or right after the values and weights
        """

        self.N = N
        self.problem_instance = self.change_problem_instance(problem_instance) if problem_instance != None else None

        # Internal tracking
        # We'll store (value, weight, idx) so we know which normalized entry to use.
        self.items: List[Tuple[float, float, int]] = []
        self.remaining_capacity = 0.0
        self.current_value = 0.0
        self.current_weight = 0.0
        self.current_weight_sum = self.total_value if problem_instance else 0
        self.current_value_sum = self.total_weight if problem_instance else 0
        self.done = False
        self.best_value = 0.0
        self.pad_meta_data_at_back=pad_meta_data_at_back

        if positive_reward_function is None: raise ValueError("Positive reward MUST be defined")
        if negative_reward_function is None: raise ValueError("Negative reward MUST be defined")
        
        self.positive_reward_function:Callable[[float, float,float], float] = positive_reward_function
        self.negative_reward_function:Callable[[float, float,float], float] = negative_reward_function

        if self.problem_instance is not None:
            self.reset()
    

    def _normalize(self, values:List[float], weights:List[float]):
        # Precompute normalized values and weights for the reward function
        # v_r_i = v_i / (w_i * capacity), w_r_i = w_i / capacity

        normalized_values = []
        normalized_weights = []
        for v, w in zip(values, weights):
            if w == 0.0:
                # Avoid division by zero; you can choose an alternate scheme
                # for zero-weight items (e.g., treat as v / (1 * capacity))
                normalized_values.append(0.0)
            else:
                normalized_values.append(v / (w * self.capacity))
            normalized_weights.append(w / self.capacity)
        
        return normalized_values, normalized_weights
    
    def change_problem_instance(self, problem_instance:dict[str, Any]) -> None:

        assert len(problem_instance['values']) ==  len(problem_instance['weights']), "Weights and Values must match"
        assert len(problem_instance['values']) != 0, "KnapsackEnv canont take empty KP instances"
        assert self.N >= len(problem_instance['values']), f"KnapsackEnv cannot take KP instances larger than {self.N}. Expected: <= {self.N}, got {len(problem_instance['values'])}"

        self.problem_instance = problem_instance

        # Extract basic info
        self.values = list(problem_instance['values'])
        self.weights = list(problem_instance['weights'])
        self.capacity = float(problem_instance['capacity'])
        self.n_items = len(self.values)

        self.total_value = sum(self.values)
        self.total_weight = sum(self.weights)

        normalized_values, normalized_weights = self._normalize(self.values, self.weights)
        self.normalized_values = normalized_values
        self.normalized_weights = normalized_weights

        # Define the action space: pick an index [0, n_items - 1]
        self.action_space = spaces.Discrete(self.n_items)

        # Observation space: 2*n for (value, weight) pairs + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self.n_items + 4,),
            dtype=np.float32
        )

        return self.problem_instance

    def reset(self) -> np.ndarray:
        """
        Reset the environment state and return the initial observation.
        """
        # Re-initialize the item list with all items + their original indices
        self.items = [
            (v, w, i) for i, (v, w) in enumerate(zip(self.values, self.weights))
        ]
        self.remaining_capacity = self.capacity
        self.current_value = 0.0
        self.current_weight = 0.0
        self.current_weight_sum = self.total_weight
        self.current_value_sum = self.total_value
        self.done = False
        # self.best_value = 0.0  # Uncomment if you want to reset best_value each episode

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Build the state vector of length (2*n_items + 4).
          - The first 2*len(self.items) slots: (value_i, weight_i) for each remaining item (left-aligned).
          - The next 2*(n_items - len(self.items)) slots: 0.0
          - The last 4 slots: [capacity, total_value, total_weight, n_items].
        """
        # state = np.zeros(2 * self.n_items + 4, dtype=np.float32)
        state = np.zeros(2 * self.N + 4, dtype=np.float32)

        # Fill in item data for remaining items (left-aligned)

        for i, (v, w, idx) in enumerate(self.items):
            offset = 0 if self.pad_meta_data_at_back else 4
            pos = offset + 2 * i
            state[pos] = self.normalized_values[idx]
            state[pos + 1] = self.normalized_weights[idx]
            
        # The last 4 features
        if self.pad_meta_data_at_back:
            state[-4] = self.capacity
            state[-3] = self.current_value_sum
            state[-2] = self.current_weight_sum
            state[-1] = len(self.items)
        else:
            state[0] = self.capacity
            state[1] = self.current_value_sum
            state[2] = self.current_weight_sum
            state[3] = len(self.items)

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action (select an item index) and return (new_state, reward, done, info).
        Reward logic:
          1) If action is out of range => reward = -W_P.
          2) Else pick the item; if it fits => reward = v_r_i; else => reward = - w_r_i.
          3) Remove the item from self.items in all cases (shifting the state).
        End the episode if no items remain or if capacity is exhausted.
        """
        # print(self.problem_instance)
        assert self.problem_instance is not None, "KP Problem Intance is None, it has not been set!"
        reward = 0.0
        item_picked = None

        # Out-of-range or invalid action => large negative penalty
        if action < 0 or action >= len(self.items):
            reward = -self.remaining_capacity
            # self.done = True
        else:
            # Retrieve the chosen item
            item_value, item_weight, item_idx = self.items[action]

            # Check if it fits in the *remaining* capacity
            if item_weight <= self.remaining_capacity:
                # Positive reward: v_r_i = v_i / (w_i * W_P)
                # reward = self.normalized_values[item_idx]
                reward = self.positive_reward_function(item_value, item_weight, self.capacity)


                # Update current knapsack usage and variables
                self.current_value += item_value
                self.current_weight += item_weight
                self.remaining_capacity -= item_weight
                self.current_weight_sum -= item_weight
                self.current_value_sum -= item_value
            else:
                # Negative reward: - wr_i = - (w_i / W_P)
                # reward = -self.normalized_weights[item_idx]
                reward = -self.negative_reward_function(item_value, item_weight, self.capacity)

            # Remove this item from the state (pop shifts everything to the left).
            item_picked = self.items.pop(action)

        # Check if we should end the episode
        if len(self.items) == 0 or self.remaining_capacity == 0:
            self.done = True
            # Update best_value if the current solution is better
            if self.current_value > self.best_value:
                self.best_value = self.current_value

        # Build the new state
        new_state = self._get_state()

        info = {
            'current_value': self.current_value,
            'current_weight': self.current_weight,
            'remaining_capacity': self.remaining_capacity,
            'best_value': self.best_value,
            'items_remaining': len(self.items),
            'item': item_picked
        }

        return new_state, reward, self.done, info

    def render(self, mode='human') -> None:
        """
        Print out the environment state for debugging.
        """
        if mode == 'human':
            print("Knapsack Environment")
            print(f"Capacity: {self.capacity}, Remaining: {self.remaining_capacity}")
            print(f"Current value: {self.current_value}, Current weight: {self.current_weight}")
            print("Items left (value, weight, idx):")
            for it in self.items:
                print("  ", it)
            print(f"Best value so far: {self.best_value}")
            print(f"Done: {self.done}")
            print("-" * 50)

        
    def getNOfItems(self) -> int:
        return self.n_items

def run_episode_test(env:KnapsackEnv, policy=None, render=False, max_ite = None):
    """
    Run a single episode in a Gym environment.

    Args:
        env: An instance of a Gym environment.
        policy: Optional function that takes the current state and returns an action.
                If None, actions are chosen randomly.
        render: Boolean flag to indicate whether to render the environment at each step.

    Returns:
        total_reward (float): The total accumulated reward from the episode.
        episode_info (dict): The final info dictionary returned by the environment.
    """
    state = env.reset()
    total_reward = 0.0
    done = False
    ite = 0
    while not done:
        if render:
            env.render()
        # Choose an action: use the policy if provided, otherwise sample randomly.
        action = policy(state) if policy is not None else env.action_space.sample()
        print("Action:" , action, "State:", state)
        state, reward, done, info = env.step(action)
        print("Reward:", reward)


        total_reward += reward
        ite += 1
        if max_ite is not None and ite < max_ite:
            break

    if render:
        env.render()

    return total_reward, info


# # Example usage:
# def main() -> None:
#     # Define a simple problem instance
#     problem_instance = {
#         'values': [60, 100, 120],
#         'weights': [10, 20, 40],
#         'capacity': 50
#     }

#     # Create the Knapsack environment (assuming KnapsackEnv is already defined)
#     env = KnapsackEnv(problem_instance, N=3)
    

#     # Run an episode using a random policy (since no policy is provided)
#     total_reward, final_info = run_episode_test(env, render=True)
#     print("Episode finished with total reward:", total_reward)
#     print("Final info:", final_info)

# main()