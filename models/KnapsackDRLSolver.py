from environment.knapsackgym import KnapsackEnv
from models.AbstractKnapsackPolicy import AbstractKnapsackPolicy
from models.StateAggregator import StateAggregator
from typing import List, Callable, Optional, Union, Tuple, Dict, Any
from models.DP_Knapsack import solve_knapsack_dp, solve_KP_instances_with_DP
import numpy as np
from models.Greedy_Knapsack import greedy_knapsack, solve_problem_instances_greedy

from numpy._typing import NDArray

class KnapsackDRLSolver:
    """DRL-based Knapsack Solver using A2C algorithm"""
    
    def __init__(self, env, KPsolver:AbstractKnapsackPolicy, use_state_aggregation=False, verbose=True):
        """
        Initialize the DRL solver.
        
        Args:
            env: Gym environment for the knapsack problem
            N (int): Maximum number of items in a problem instance
            use_state_aggregation (bool): Whether to use state aggregation
            gamma (float): Discount factor
            lr_policy (float): Learning rate for policy network
            lr_value (float): Learning rate for value network
        """
        self.env:KnapsackEnv = env
        self.N = KPsolver.N
        self.use_state_aggregation = use_state_aggregation
        
        # Initialize state aggregator if needed
        self.state_aggregator = StateAggregator(self.N) if use_state_aggregation else None
        self.KPsolver = KPsolver
        self.verbose = verbose
        
    def process_state(self, state, P_idx=None):
        """
        Process state with optional aggregation.
        
        Args:
            state (numpy.ndarray): Original state
            P_idx (int): Problem index
            
        Returns:
            numpy.ndarray: Processed state
        """
        if self.use_state_aggregation and self.state_aggregator is not None:
            return self.state_aggregator.aggregate(state, P_idx)
        return state
    
    def train(self, problem_instances, t_max=None, param_update_ticker=1):
        """
        Train the DRL solver on multiple problem instances with progress tracking.
        
        Args:
            problem_instances (List[Dict]): List of problem instances
            t_max (int): Maximum number of training steps
            
        Returns:
            Dict: Training data including:
                - 'instance_best_values': Final best values for each problem instance
                - 'best_values_over_time': 2D array of best values for each instance over time
                - 'best_sum_over_time': Sum of best values across all instances over time
                - 'avg_rewards_over_time': Average rewards per episode over time
                - 'param_update_ticker': how often to update model parameters
        """
        assert len(problem_instances) is not None or len(problem_instances) > 0

        if t_max is None:
            t_max = 3 * self.N * 10000  # As specified in the pseudocode

        if self.use_state_aggregation:
            self.state_aggregator.train(problem_instances, self.N)
        
        # Initialize tracking variables
        val = np.zeros(len(problem_instances))  # Best values for each problem instance
        val_sum = 0 # Keep tracks of the total sum of the val array

        best_values_over_time = np.zeros((t_max, len(problem_instances)))  # Track best values per instance over time
        best_sum_over_time = np.zeros(t_max)  # Track sum of best values over time
        avg_rewards_over_time = np.zeros(t_max)  # Track average rewards over time
        
        print(f"Training on {len(problem_instances)} KP Instances, with N={self.N}, t_max={t_max}")
        for t in range(t_max):
            # Select a problem instance (line 6 in pseudocode)
            # P_idx = np.random.randint(0, len(problem_instances))
            P_idx = t % len(problem_instances)
            P = problem_instances[P_idx]
            
            assert len(P['values']) <= self.N, f"Problem Instance has too many items. KnapsackEnv is configuered to accept no more than <= {self.N}."

            # Change the environment to use this problem instance
            self.env.change_problem_instance(P)
            
            # Reset environment and get initial state
            state = self.env.reset()
            
            # Initialize for this episode
            done_ = False
            ow = 0  # Total weight of selected items
            ov = 0  # Total value of selected items
            
            # Create a copy of the problem instance P for modification
            n_P_prime = len(P['values'])
            W_P_prime = P['capacity']
            
            # Store states, actions, rewards for batch update
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            # Track episode rewards
            episode_rewards = []

            total_episode_reward_sum = 0
            
            # Solve the knapsack problem for this instance
            while not done_:
                # Process state if needed
                processed_state = self.process_state(state, P_idx)
                states.append(processed_state)
                
                # Get available actions (indices of remaining items)
                available_actions = list(range(len(self.env.items)))
                
                # Get action according to policy (line 12 in pseudocode)
                action = self.KPsolver.get_action(processed_state, available_actions)
                actions.append(action)
                
                # Take action and observe reward and next state
                next_state, reward, done, info = self.env.step(action)
                total_episode_reward_sum += reward
                rewards.append(reward)
                episode_rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                # Update the current value
                ov = info['current_value']
                
                # Update state
                state = next_state

                # Update flag
                done_ = done 
            
            # Update parameters using collected trajectories
            if t % param_update_ticker == 0:
                self.KPsolver.update_parameters(states, actions, rewards, next_states, dones)
            
            # Calculate average reward for this episode
            avg_episode_reward = total_episode_reward_sum / len(episode_rewards) if episode_rewards else 0
            avg_rewards_over_time[t] = avg_episode_reward
            
            # if self.verbose and (t % 1000) == t % len(problem_instances):
            if self.verbose and (t % 1000) == 0:
                print(f"Iteration [{t}/{t_max}], Training KP Instance {P_idx}, Reward: {avg_episode_reward}")

            # Update best value if needed (lines 20-22 in pseudocode)
            if ov > val[P_idx]:
                val_sum += ov - val[P_idx]
                val[P_idx] = ov
            
            # Store the current best values for all instances for this timestep
            best_values_over_time[t] = val.copy()
            
            # Calculate and store the sum of best values across all instances
            best_sum_over_time[t] = val_sum
            # best_sum_over_time[t] = sum(val)
        
        # Prepare the return dictionary with all training data
        training_data = {
            'instance_best_values': val,  # Final best values for each instance
            'best_values_over_time': best_values_over_time,  # 2D array of best values over time
            'best_sum_over_time': best_sum_over_time,  # Sum of best values over time
            'avg_rewards_over_time': avg_rewards_over_time  # Average rewards per episode over time
        }
            
        return training_data
    
    def solve(self, problem_instance):
        """
        Solve a single knapsack problem instance using the trained policy.
        
        Args:
            problem_instance (Dict): A problem instance
            
        Returns:
            Tuple[float, List[int]]: Total value and list of selected item indices
        """
        # Set environment to use this problem instance
        self.env.change_problem_instance(problem_instance)
        
        # Reset environment and get initial state
        state = self.env.reset()
        
        done = False
        total_value = 0
        total_weight = 0
        selected_items = []
        
        while not done:
            # Process state if needed
            processed_state = self.process_state(state)
            
            # Get available actions (indices of remaining items)
            available_actions = list(range(len(self.env.items)))
            
            if not available_actions:
                break
                
            # Get action according to policy
            action = self.KPsolver.get_action(processed_state, available_actions)
            
            # Take action and observe reward and next state
            next_state, reward, done, info = self.env.step(action)
            
            # If item was added (positive reward means item fit)
            if reward > 0:
                assert info['item'] != None
                selected_items.append(info['item'][2])
            
            # Update value and weight
            total_value = info['current_value']
            total_weight = info['current_weight']
            
            # Update state
            state = next_state
        
        return total_value, total_weight, selected_items

def train_knapsack_solver(env, problem_instances:List[Dict[str, Any]], KPsolver, use_state_aggregation=False, t_max=None, verbose=True):
    """
    Train a DRL-based knapsack solver on multiple problem instances.
    
    Args:
        env: Gym environment for knapsack problem
        problem_instances (List[Dict]): List of problem instances
        N (int): Maximum number of items in a problem instance
        use_state_aggregation (bool): Whether to use state aggregation
        gamma (float): Discount factor
        lr_policy (float): Learning rate for policy network
        lr_value (float): Learning rate for value network
        t_max (int): Maximum number of training steps
        
    Returns:
        Tuple[KnapsackDRLSolver, List[float]]: Trained solver and solution values
    """
    # Initialize solver
    solver = KnapsackDRLSolver(
         env=env,
        KPsolver=KPsolver,
        use_state_aggregation=use_state_aggregation,
        verbose=verbose
    )
    
    # Train solver
    solution_values = solver.train(problem_instances, t_max)

    return solver, solution_values


def evaluate_knapsack_solver(solver:KnapsackDRLSolver, test_instances:list[dict]):
    """
    Evaluate the trained solver on test instances.
    
    Args:
        solver (KnapsackDRLSolver): Trained solver
        test_instances (List[Dict]): List of test problem instances
        
    Returns:
        List[Dict]: Evaluation results for each test instance
    """
    results = []
    
    for i, instance in enumerate(test_instances):
        # Solve instance
        value, weight, selected_items = solver.solve(instance)
        
        # Calculate actual total weight
        total_weight = sum(instance['weights'][idx] for idx in selected_items)
        
        # Check if solution respects capacity constraint
        is_valid = total_weight <= instance['capacity']

        optimal_value, optimal_items = solve_knapsack_dp(instance)
        
        # Store results
        results.append({
            'instance_idx': i,
            'total_value': value,
            'total_weight': total_weight,
            'capacity': instance['capacity'],
            'selected_items': selected_items,
            'is_valid': is_valid,
            "optimal_sol": optimal_value,
            "optimal_items": optimal_items,
            "optimality_ratio": value /  optimal_value 
        })
    
    return results

def run_KPSolver(env:KnapsackEnv, KPSolver:KnapsackDRLSolver, 
                 training_problem_instances: List[Dict[str, Any]], t_max:int=None, 
                 use_state_aggregation:bool=False, verbose:bool=True) -> Tuple[KnapsackDRLSolver, Dict[str, NDArray[np.float64]]]:
    
    print(f"Running Model {KPSolver.__class__}")
    
    # Train solver
    solver, solution_values = train_knapsack_solver(
        env=env,
        problem_instances=training_problem_instances,
        KPsolver=KPSolver,
        use_state_aggregation=use_state_aggregation,
        t_max=t_max,
        verbose=verbose
    )
    
    return solver, solution_values


def print_results(results:list):
    print("Evaluation results:")
    for res in results: 
        print(res)
