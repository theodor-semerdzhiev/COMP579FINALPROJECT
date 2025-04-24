from models.AbstractKnapsackPolicy import AbstractKnapsackPolicy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Union, Dict, Callable, Tuple
import random

# Reuse your custom networks or define them similarly if needed:
# --------------------------------------------------------------

class PolicyNetworkPPO(nn.Module):
    """
    Policy (Actor) network for PPO.
    Produces logits from which action probabilities are derived.
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 2, 
        activation: str = "sigmoid"
    ):
        super(PolicyNetworkPPO, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.activation_funcs: Dict[str, Callable] = {
            "sigmoid": F.sigmoid, 
            "relu": F.relu, 
            "tanh": F.tanh, 
            "leaky_relu": F.leaky_relu
        }

        if activation not in self.activation_funcs:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        self.activation = self.activation_funcs[activation]
        
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns raw logits for each possible action."""
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        # Last layer (logits)
        logits = self.layers[-1](x)
        return logits
    
    def get_action(self, 
                   state: Union[List[float], torch.Tensor, np.ndarray], 
                   available_actions: List[int]) -> int:
        """
        Get action according to the network's policy (softmax), 
        masking out unavailable actions as in the A2C approach.
        """
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
            else:
                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state

            logits = self.forward(state_tensor)
            action_probs = F.softmax(logits, dim=1)

            # Mask unavailable actions
            action_mask = torch.zeros_like(action_probs)
            for action in available_actions:
                if action < action_mask.shape[1]:
                    action_mask[0, action] = 1

            masked_probs = action_probs * action_mask

            # Turn any NaNs into zeros
            masked_probs = torch.nan_to_num(masked_probs, nan=0.0, posinf=0.0, neginf=0.0)

            # If all actions are masked, choose randomly
            if torch.sum(masked_probs) == 0:
                return random.choice(available_actions)

            # masked_probs = torch.clamp(masked_probs, min=1e-10)  # Remove zeros/negatives
            total = masked_probs.sum()
            # if not torch.isfinite(total).all():
            #     print(masked_probs, torch.sum(masked_probs), state_tensor, logits, action_probs, action_mask)

            if total.item() == 0.0:
                # index of allowed actions
                allowed = masked_probs.nonzero(as_tuple=False).squeeze(1)
                # pick one uniformly at random
                return int(allowed[torch.randint(len(allowed), (1,))])
            
            masked_probs_ = masked_probs / total # Renormalize to sum to 1

            # if not torch.isfinite(masked_probs).all():
            #     # print("Non-finite probs:", masked_probs[~torch.isfinite(masked_probs)])
            #     print(total, masked_probs_, masked_probs)

            action = torch.multinomial(masked_probs_, 1).item()


            return action
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probabilities of given actions under the current policy.
        Also returns the entropy of the distribution for regularization.
        """
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # For entropy computation, we'll just do a standard softmax => 
        probs = F.softmax(logits, dim=1)
        dist_entropy = -torch.sum(probs * log_probs, dim=1).mean()

        return action_log_probs, dist_entropy


class ValueNetworkPPO(nn.Module):
    """
    Value (Critic) network for PPO, outputting a single scalar value for the state.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 2, 
        activation: str = "sigmoid"
    ):
        super(ValueNetworkPPO, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.activation_functions: Dict[str, Callable] = {
            "sigmoid": F.sigmoid,
            "relu": F.relu,
            "tanh": F.tanh,
            "leaky_relu": F.leaky_relu
        }
        
        if activation not in self.activation_functions:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        self.activation = self.activation_functions[activation]
        
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        # Output layer
        self.layers.append(nn.Linear(hidden_size, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        value = self.layers[-1](x)
        return value.squeeze(-1)  # Return shape [batch], removing last dim


class KnapsackPPOSolver(AbstractKnapsackPolicy):
    """
    Knapsack solver using Proximal Policy Optimization (PPO).
    Inherits from AbstractKnapsackPolicy so it can be used 
    directly in your existing DRL pipeline for 0-1 Knapsack.
    """
    def __init__(
        self, 
        N: int = 100, 
        gamma: float = 0.99, 
        policy_lr: float = 0.001, 
        value_lr: float = 0.001,
        clip_epsilon: float = 0.2,
        K_epochs: int = 5,
        entropy_coef: float = 0.01,
        activation: str = "sigmoid",
        hidden_size: int = 64,
        num_layers: int = 2,
        verbose: bool = True
    ):
        """
        Args:
            N (int): Max number of items
            gamma (float): Discount factor
            lr_policy (float): Learning rate for the policy network
            lr_value (float): Learning rate for the value network
            clip_epsilon (float): Clipping parameter epsilon for PPO
            K_epochs (int): Number of epochs (updates) per PPO batch
            entropy_coef (float): Weight for the entropy bonus
            activation (str): Activation function for both actor and critic
            hidden_size (int): Number of hidden units in each layer
            num_layers (int): Number of hidden layers
            verbose (bool): Whether to print log messages during training
        """
        super().__init__(N=N, gamma=gamma)
        self.N = N
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.verbose = verbose

        # Define the network sizes
        input_size = 2 * N + 4
        output_size = N
        
        # Initialize the Actor (policy) and Critic (value)
        self.policy_net = PolicyNetworkPPO(
            input_size=input_size, 
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation
        )
        self.value_net = ValueNetworkPPO(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation
        )

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)

    def get_action(self, 
                   state: Union[List[float], torch.Tensor, np.ndarray], 
                   available_actions: List[int]) -> int:
        """
        Select an action from the policy network, masking unavailable actions.
        This method is called by your environment-solving code.
        """
        return self.policy_net.get_action(state, available_actions)

    def update_parameters(self, 
                          states: List[np.ndarray], 
                          actions: List[int], 
                          rewards: List[float], 
                          next_states: List[np.ndarray], 
                          dones: List[bool]) -> None:
        """
        Perform a PPO update given a single trajectory (or batch of trajectories):
          1) Compute discounted returns.
          2) Compute advantages (returns - baseline).
          3) Run multiple epochs of clipped PPO updates on this batch.
        """
        # Convert everything to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)

        # Compute the value of current states and next states
        with torch.no_grad():
            state_values = self.value_net(states_tensor)
            next_state_values = self.value_net(next_states_tensor)

        # --- 1) Compute discounted returns ---
        returns = []
        R = 0.0
        for i in reversed(range(len(rewards))):
            # If done, reset R to 0 so next episode is not mixed
            if dones[i]:
                R = 0.0
            R = rewards[i] + self.gamma * R
            returns.insert(0, R)

        returns_tensor = torch.FloatTensor(returns)

        # --- 2) Compute advantage estimates: advantage = returns - baseline
        advantages = returns_tensor - state_values.detach()

        # Evaluate old log probs with the *current* policy params
        # (In standard PPO, you'd store log_probs from the old policy; 
        #  but we can re-calculate them from the snapshot of the "old policy" 
        #  if we had saved it. For simplicity, we just compute them now.)
        with torch.no_grad():
            old_log_probs, _ = self.policy_net.evaluate_actions(states_tensor, actions_tensor)

        # --- PPO Update loop (K_epochs) ---
        for _ in range(self.K_epochs):
            # 2.1) Evaluate log_probs under *new* policy
            log_probs, dist_entropy = self.policy_net.evaluate_actions(states_tensor, actions_tensor)

            # 2.2) Compute ratio = exp(new_log_prob - old_log_prob)
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO's Clipped Surrogate objective:
            # unclipped = ratio * advantage
            # clipped = clamp(ratio, 1-eps, 1+eps) * advantage
            # objective = - mean( min(unclipped, clipped) )
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # Add optional entropy bonus
            entropy_loss = -dist_entropy * self.entropy_coef

            # Combine total policy loss
            total_policy_loss = policy_loss + entropy_loss

            # Backprop for policy
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            self.policy_optimizer.step()

            # Update value network
            new_state_values = self.value_net(states_tensor)
            value_loss = F.mse_loss(new_state_values, returns_tensor)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        if self.verbose:
            avg_return = returns_tensor.mean().item()
            avg_val_loss = value_loss.item()
            avg_policy_loss = policy_loss.item()
            print(
                f"[PPO update] AvgReturn: {avg_return:.3f}, "
                f"PolicyLoss: {avg_policy_loss:.4f}, ValueLoss: {avg_val_loss:.4f}"
            )

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
