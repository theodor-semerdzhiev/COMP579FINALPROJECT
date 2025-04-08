import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Callable, Union
import random
from AbstractKnapsackPolicy import AbstractKnapsackPolicy

class PolicyNetwork(nn.Module):
    """Policy network for the A2C algorithm to solve the Knapsack Problem"""
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 2, 
        activation: str = "sigmoid"
    ):
        """
        Initialize the policy network with customizable architecture.
        
        Args:
            input_size (int): Size of the input vector (2N + 4)
            output_size (int): Size of the output vector (N)
            hidden_size (int): Size of the hidden layers (default: 64)
            num_layers (int): Number of hidden layers (default: 2)
            activation (str): Activation function to use (default: "sigmoid")
                              Options: "sigmoid", "relu", "tanh", "leaky_relu"
        """
        super(PolicyNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.activation_funcs:Dict[str, Callable] = {
            "sigmoid":F.sigmoid, 
            "relu": F.relu, 
            "tanh": F.tanh, 
            "leaky_relu": F.leaky_relu
        }

        # Set activation function
        if activation not in self.activation_funcs:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        self.activation = self.activation_funcs[activation]
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        # Pass through all layers except the last one with activation
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        
        # No activation on the output layer
        return self.layers[-1](x)
    
    def get_action(self, state: Union[List[float], torch.Tensor, np.ndarray], 
                  available_actions: List[int]) -> int:
        """
        Get action according to policy.
        
        Args:
            state (Union[List[float], torch.Tensor, numpy.ndarray]): Current state
            available_actions (List[int]): List of available actions
            
        Returns:
            int: Chosen action index
        """
        with torch.no_grad():
            # Convert state to tensor if it's not already
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
            else:
                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
                
            action_probs = F.softmax(self.forward(state_tensor), dim=1)
            
            # Mask unavailable actions
            action_mask = torch.zeros_like(action_probs)
            for action in available_actions:
                if action < action_mask.shape[1]:
                    action_mask[0, action] = 1
            
            masked_probs = action_probs * action_mask
            
            # If all actions are masked, choose randomly from available actions
            if torch.sum(masked_probs) == 0:
                return random.choice(available_actions)
            
            # Normalize probabilities
            masked_probs = masked_probs / torch.sum(masked_probs)
            
            # Sample action from the masked probability distribution
            action = torch.multinomial(masked_probs, 1).item()
            
            return action
        


class ValueNetwork(nn.Module):
    """Value network for the A2C algorithm to estimate state values"""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 2, 
        activation: str = "sigmoid"
    ):
        """
        Initialize the value network with customizable architecture.
        
        Args:
            input_size (int): Size of the input vector (2N + 4)
            hidden_size (int): Size of the hidden layers (default: 64)
            num_layers (int): Number of hidden layers (default: 2)
            activation (str): Activation function to use (default: "sigmoid")
                             Options: "sigmoid", "relu", "tanh", "leaky_relu"
        """
        super(ValueNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Dictionary mapping activation function names to their implementations
        self.activation_functions: Dict[str, Callable] = {
            "sigmoid": F.sigmoid,
            "relu": F.relu,
            "tanh": F.tanh,
            "leaky_relu": F.leaky_relu
        }
        
        # Set activation function
        if activation not in self.activation_functions:
            raise ValueError(f"Unsupported activation function: {activation}. "
                            f"Supported options are: {list(self.activation_functions.keys())}")
        
        self.activation = self.activation_functions[activation]
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            
        # Output layer - value networks output a single scalar value
        self.layers.append(nn.Linear(hidden_size, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Estimated state value
        """
        # Pass through all layers except the last one with activation
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        
        # No activation on the output layer for value estimation
        return self.layers[-1](x)


class KnapsackA2C(AbstractKnapsackPolicy):
    def __init__(self, N=100, gamma=0.99, lr_policy=0.001, lr_value=0.001, verbose=True):
        super().__init__(N=N, gamma=gamma)
        self.N = N
        self.gamma = gamma
        
        # Define input and output sizes for networks
        input_size = 2 * N + 4  # As specified in the requirements
        output_size = N  # One output per potential item
        
        # Initialize networks
        self.policy_net = PolicyNetwork(input_size, output_size)
        self.value_net = ValueNetwork(input_size)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)

        self.verbose = verbose
    
    def get_action(self, state: Union[List[float], torch.Tensor, np.ndarray], 
                  available_actions: List[int]) -> int:
        return self.policy_net.get_action(state, available_actions)

    def update_parameters(self, states, actions, rewards, next_states, dones):
        """
        Update policy and value network parameters using collected trajectories.
        
        Args:
            states (List[numpy.ndarray]): List of states
            actions (List[int]): List of actions
            rewards (List[float]): List of rewards
            next_states (List[numpy.ndarray]): List of next states
            dones (List[bool]): List of done flags
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        
        # Compute state values
        state_values = self.value_net(states_tensor).squeeze()
        next_state_values = self.value_net(next_states_tensor).squeeze()
        
        # Compute returns and advantages for A2C
        returns = []
        advantages = []
        R = 0
        
        # Compute returns with bootstrapping
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
        
        returns_tensor = torch.FloatTensor(returns)
        
        # Compute advantage = returns - state_values
        advantages = returns_tensor - state_values.detach()
        
        # Update value network (equation 3 in pseudocode)
        value_loss = F.mse_loss(state_values, returns_tensor)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy network (equation 2 in pseudocode)
        policy_output = self.policy_net(states_tensor)
        action_probs = F.softmax(policy_output, dim=1)
        
        # Extract probabilities of chosen actions
        action_log_probs = F.log_softmax(policy_output, dim=1)
        selected_action_log_probs = action_log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        
        # Policy gradient loss
        policy_loss = -(selected_action_log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
    def save(self, path: str) -> None:
        # Implementation of the optional method
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        # Implementation of the optional method
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        
    
