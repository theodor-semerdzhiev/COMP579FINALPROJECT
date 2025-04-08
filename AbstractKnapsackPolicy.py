from typing import List, Union
import numpy as np
import torch
from abc import ABC, abstractmethod


class AbstractKnapsackPolicy(ABC):
    """
    Abstract base class for Knapsack DRL solvers.
    Any concrete implementation must implement all abstract methods.
    """
    
    def __init__(self, N: int = 100, gamma: float = 0.99):
        """
        Initialize the abstract knapsack solver.
        
        Args:
            N (int): Size parameter for the knapsack problem (e.g., max number of items)
            gamma (float): Discount factor for future rewards
        """
        self.N = N
        self.gamma = gamma
    
    @abstractmethod
    def get_action(self, state: Union[List[float], torch.Tensor, np.ndarray], 
                  available_actions: List[int]) -> int:
        """
        Select an action based on the current state and available actions.
        
        Args:
            state: The current state representation
            available_actions: List of available actions
            
        Returns:
            int: The selected action
        """
        pass
    
    @abstractmethod
    def update_parameters(self, states: List, actions: List[int], 
                         rewards: List[float], next_states: List, 
                         dones: List[bool]) -> None:
        """
        Update policy and value network parameters using collected trajectories.
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
        """
        raise NotImplementedError("Saving functionality must be implemented in derived class")
    
    def load(self, path: str) -> None:
        """
        Load the model from the specified path.
        
        Args:
            path: Path to load the model from
        """
        raise NotImplementedError("Loading functionality must be implemented in derived class")