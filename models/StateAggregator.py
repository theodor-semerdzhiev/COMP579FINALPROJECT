import numpy as np
import random
import math

class StateAggregator:
    """
    State aggregator to reduce state space dimension.
    See section 4.2 in the paper for the description
    """
    
    def __init__(self, max_splits):
        """
        Initialize the state aggregator

        Args:
            max_splits (int): Maximum number of splits for the aggregator
        """
        self.max_splits = max_splits
        
    def aggregate(self, state, P_idx):
        """
        Aggregate state according to equation (7) as mentioned in pseudocode.
        
        Args:
            state (numpy.ndarray): Original state
            P_idx (int): problem set id to find the aggregated state in the map. If this is a test problem set, no processing is done.
            
        Returns:
            numpy.ndarray: Aggregated state
        """
        if P_idx is None:
            return state    # state was not aggregated in training, no processing is done

        assert self.state_map is not None   # cannot call the aggregate function without training first

        # initialize state space properties
        capacity = state[-4]
        total_value = 0
        total_weight = 0
        n_items = state[-1]

        # iterate through the state and map original the values and weights to the aggregated ones
        new_state = list()
        for i in range((len(state) - 4) // 2):
            new_state += [self.state_map[P_idx, i, 1]]
            new_state += [self.state_map[P_idx, i, 2]]
            total_value += self.state_map[P_idx, i, 1]
            total_weight += self.state_map[P_idx, i, 2]

        # add state space properties to the vector
        new_state += [capacity]
        new_state += [total_value]
        new_state += [total_weight]
        new_state += [n_items]

        # return aggregated space
        return new_state
    
    def train(self, problem_instances, N, alpha=0.9, gamma=0.3, epsilon=1, decay=0.9):
        """
        Find aggregation strategy for all problem sets according to algorithm 2

        Args:
            problem_instances (List[Dict]): List of problem instances
            N (int): Maximum number of items in a problem
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Used in the epsilon-greedy action selection
            decay (float): How much the learning rate and discount factor decrease each cycle

        Returns:
            None
        """

        # Create a table of features from all problem instances
        # Axis 0 - problem instance, Axis 1 - item index, Axis 2 - 4 tuple of (item index, value, weight, problem set index)
        self.features = list()
        for k, instance in enumerate(problem_instances):
            row = list()
            for i in range(N):
                # Non-existant values and weights are filled in with 0s
                if i >= len(instance["values"]):
                    row += [[i,0,0,k]]
                else:
                    row += [[i, instance["values"][i], instance["weights"][i], k]]
            self.features += [row]

        # Each feature vector (v_i for all problems) is sorted in ascending order for the aggregator
        self.features = np.array(self.features)
        column_feats = np.transpose(self.features, (1, 0, 2))
        column_feats = np.array([sorted(row, key=lambda r: r[1]) for row in column_feats])
        sorted_feats = np.transpose(column_feats, (1, 0, 2))

        self.N = N

        # Performs Q-learning to determine the optimal split number for each feature vector
        splits = self.q_learning(sorted_feats, alpha, gamma, epsilon, decay)

        # From each split, determine the bin sizes
        aggregator = np.vectorize(lambda i : math.ceil(len(problem_instances) / (i + 1)))
        bin_sizes = aggregator(splits)

        # constructs map from original features to aggregated ones
        self.state_map = np.zeros(sorted_feats.shape)
        for i in range(N):
            for j in range(len(problem_instances)):
                self.state_map[j,i,0] = sorted_feats[j,i,0]                     # i
                self.state_map[j,i,1] = j // bin_sizes[i]                       # bin id
                self.state_map[j,i,2] = self.aggr_weight(sorted_feats[j,i,2])   # weight
                self.state_map[j,i,3] = sorted_feats[j,i,3]                     # original problem set
        
        # sorts the features back into their original order
        column_feats = np.transpose(self.state_map, (1, 0, 2))
        column_feats = np.array([sorted(row, key=lambda r: r[3]) for row in column_feats])
        self.state_map = np.transpose(column_feats, (1, 0, 2))


    def aggr_weight(self, weight):
        """
        Helper function that determines the weight of an aggregated value

        Args:
            weight (float): The input weight
        
        Returns:
            int: Aggregated weight
        """
        if weight <= 0.5:
            return 0
        elif weight <= 1:
            return 1
        return 2

    def q_learning(self, features, alpha, gamma, epsilon, decay):
        """
        Performs Q-Learning to find the best number of splits, see Algorithm 2 in the paper

        Args:
            features (ndarray(float)): 3D array of input features
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Used in the epsilon-greedy action selection
            decay (float): How much the learning rate and discount factor decrease each cycle
        
        Returns:
            ndarray(int): Number of splits
        """
        # Threshold for convergence
        c_threshold = 0.001

        # Create a Q table
        Q = np.zeros((self.N, self.max_splits))

        # Iterate until convergence
        cum_delta = 1
        i = random.randint(0, self.N - 1)
        while cum_delta > c_threshold:
            i_prime = random.randint(0, self.N - 1)
            a = self.e_greedy(Q, i, epsilon)
            r = self.reward(i, a, features)
            delta = alpha * (r + (gamma * np.max(Q[i_prime, :])) - Q[i, a])
            Q[i, a] += delta
            
            cum_delta = decay * cum_delta + (1 - decay) * abs(delta)
            i = i_prime
            alpha *= decay
            epsilon *= decay

        # Return the splits with highest Q values
        return np.argmax(Q, axis=0)
    
    def reward(self, state, action, features):
        """
        Helper function that determines the reward associated to a feature and proposed split value. See equation 6 in the paper.

        Args:
            state (int): The item ID
            action (int): Proposed split value
            features (ndarray(float)): array of features
        
        Returns:
            float: Reward
        """
        bucket_size = math.ceil(self.features.shape[0] / (action + 1))
        top = (bucket_size ** action) * (self.features.shape[0] - (bucket_size * action))
        common_vals = self.check_common(state, bucket_size, features)
        bottom = (action + 1) * (common_vals + 1)
        return top / bottom
    
    def e_greedy(self, Q, i, epsilon):
        """
        Helper function that performs epsilon-greedy action selection

        Args:
            Q (ndarray(float)): The Q table
            i (int): Current state
            epsilon (float): Probability of choosing the best action over a random one
        
        Returns:
            int: Action
        """
        if epsilon > random.random():
            return np.argmax(Q[i, :])
        else:
            return random.randint(0, Q.shape[1] - 1)
    

    def check_common(self, feature, split_size, features):
        """
        Helper function that checks for the number of common values across bins

        Args:
            feature (int): Feature vector to check
            split_size (int): Number of splits
            features (ndarray): Array of features
        
        Returns:
            int: Common values
        """
        values = dict()
        seen = dict()

        # Records the values that are seen in each bin. Increment counter in values for every time a value occurs in a bin
        for i in range(features.shape[0]):
            if i % split_size == 0:
                seen = dict()
            if features[i, feature, 1] not in seen:
                if features[i, feature, 1] not in values:
                    values[features[i, feature, 1]] = 1
                else:
                    values[features[i, feature, 1]] += 1
            seen[features[i, feature, 1]] = True
        return sum(values.values()) - len(values)   # only count repeating values