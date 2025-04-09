import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from typing import List, Union, Tuple, Any
from AbstractKnapsackPolicy import AbstractKnapsackPolicy
from collections import deque


class QNetwork(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 2, 
        activation: str = "relu"
    ):
        """
        Q-Network that outputs Q-values for each action (of size output_size).
        Args:
            input_size (int): Dimension of the input state (2*N + 4).
            output_size (int): Dimension of the output (N).
            hidden_size (int): Hidden layer size.
            num_layers (int): Number of hidden layers.
            activation (str): Activation function: "relu", "tanh", "sigmoid", or "leaky_relu".
        """
        super(QNetwork, self).__init__()
        
        self.activation_funcs = {
            "relu": F.relu,
            "tanh": F.tanh,
            "sigmoid": F.sigmoid,
            "leaky_relu": F.leaky_relu
        }
        
        if activation not in self.activation_funcs:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        self.activation = self.activation_funcs[activation]
        
        # Create layers: input -> hidden... -> hidden -> output
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            in_dim = hidden_size
        # Final layer => output_size
        layers.append(nn.Linear(in_dim, output_size))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns Q-values for each possible action."""
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        # Last layer (no activation, we want raw Q-values)
        x = self.layers[-1](x)
        return x



class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        A simple replay buffer that stores transitions in a deque.
        Args:
            capacity (int): The maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a single transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class KnapsackDQN(AbstractKnapsackPolicy):
    def __init__(
        self, 
        N: int = 100, 
        gamma: float = 0.99, 
        lr: float = 1e-3,
        hidden_size: int = 64, 
        num_layers: int = 2, 
        activation: str = "sigmoid",
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 1e-4,
        target_update_freq: int = 1000,
        verbose: bool = True
    ):
        """
        Deep Q-Network approach for 0-1 Knapsack, following AbstractKnapsackPolicy.
        
        Args:
            N (int): Maximum number of items. 
            gamma (float): Discount factor.
            lr (float): Learning rate for the Q-network.
            hidden_size (int): Number of hidden units in each layer.
            num_layers (int): Number of hidden layers.
            activation (str): Activation function ("relu", "tanh", "sigmoid", "leaky_relu").
            buffer_capacity (int): Capacity of the replay buffer.
            batch_size (int): Mini-batch size for training updates.
            epsilon_start (float): Initial epsilon for epsilon-greedy.
            epsilon_end (float): Minimum epsilon.
            epsilon_decay (float): Epsilon decay rate per step.
            target_update_freq (int): Number of updates between target network syncs.
            verbose (bool): Whether to print logs.
        """
        super().__init__(N=N, gamma=gamma)
        
        self.N = N
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.verbose = verbose

        # Q-Network and Target network
        input_size = 2 * N + 4
        output_size = N
        self.q_net = QNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation
        )
        self.target_net = QNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation
        )
        # Sync target_net params
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # Target net in eval mode

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Update counter
        self.update_count = 0
        
    def get_action(self, 
                   state: Union[List[float], np.ndarray, torch.Tensor], 
                   available_actions: List[int]) -> int:
        """
        Choose an action using epsilon-greedy strategy, 
        masking out unavailable actions.
        """
        # Convert state to tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        state = state.unsqueeze(0)  # shape: [1, input_size]

        # Epsilon-greedy
        if random.random() < self.epsilon:
            # Random action among available
            action = random.choice(available_actions)
        else:
            # Greedy action from Q-Net among available
            with torch.no_grad():
                q_values = self.q_net(state)  # shape: [1, N]
                # Mask unavailable actions by setting Q to a large negative
                mask = torch.full_like(q_values, float('-inf'))
                for a in available_actions:
                    mask[0, a] = q_values[0, a]
                # Choose max
                action = mask.argmax(dim=1).item()

        return action

    def update_parameters(self, 
                          states: List[np.ndarray], 
                          actions: List[int], 
                          rewards: List[float], 
                          next_states: List[np.ndarray], 
                          dones: List[bool]) -> None:
        """
        Store transitions in replay buffer and run DQN updates (if buffer is large enough).
        Each call is presumably from a single episode or partial trajectory,
        but your design can vary (batch them, etc.).
        """
        # 1) Store transitions in replay buffer
        for i in range(len(states)):
            self.replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # 2) Decrease epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        # 3) If buffer is large enough, do an update step
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        # Sample mini-batch
        states_b, actions_b, rewards_b, next_states_b, dones_b = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.FloatTensor(states_b)
        actions_t = torch.LongTensor(actions_b)
        rewards_t = torch.FloatTensor(rewards_b)
        next_states_t = torch.FloatTensor(next_states_b)
        dones_t = torch.FloatTensor(dones_b)

        # 4) Compute current Q(s,a)
        q_values = self.q_net(states_t)  # shape: [batch_size, N]
        # Gather the Q-value corresponding to chosen actions
        q_action = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # shape: [batch_size]

        # 5) Compute target = r + gamma * max Q_target(s', .) * (1 - done)
        with torch.no_grad():
            # Q-target for next states
            next_q_values = self.target_net(next_states_t)  # shape: [batch_size, N]
            next_q_max = next_q_values.max(dim=1)[0]  # shape: [batch_size]
            target = rewards_t + (1 - dones_t) * self.gamma * next_q_max

        # 6) Compute loss (MSE or Smooth L1)
        loss = F.mse_loss(q_action, target)

        # 7) Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        # 8) Periodically update target network
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.verbose and self.update_count % 100 == 0:
            print(f"[DQN Update #{self.update_count}] Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.3f}")

    def save(self, path: str) -> None:
        """
        Save the Q-network and target network.
        """
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """
        Load the Q-network and target network.
        """
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
