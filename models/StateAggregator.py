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
            
        Returns:
            numpy.ndarray: Aggregated state
        """
        # For now, simply return the original state (no aggregation)

        # TODO
        # iterate through state, use i/2 as index, replace values and weights based on aggregation
        # if index not set, put it in closest bin? for now just reject
        assert P_idx is not None
        return state
    
    def train(self, problem_instances, N, alpha, gamma, epsilon, decay):
        """
        Find aggregation strategy for all problem sets according to algorithm 2

        Args:
            problem_instances (List[Dict]): List of problem instances
            N (int): Maximum number of items in a problem
        
        Returns:
            None
        """

        # Create a table of features
        self.features = list()
        for instance in problem_instances:
            row = list()
            for i, value in enumerate(instance["values"]):
                row += [[i, value, instance["weights"][i]]]
            self.features += row
        
        sorted(self.features, key = lambda row: row[0][1]) # misunderstood this, will be sorted for every i and moved to qlearn
        self.features = np.array(self.features)

        self.N = N

        splits = self.q_learning(alpha, gamma, epsilon, decay)

        # TODO
        # split weights, values
        # sort rows back into order

    def split(self, features, index, splits):
        return features #modify features in place given policy


    def q_learning(self, alpha, gamma, epsilon, decay):
        """
        Performs Q-Learning to find the best number of splits

        Args:

        
        Returns:
            int: Number of splits
        """
        c_threshold = 0.001

        # Create a Q table
        Q = np.zeros((self.N, self.max_splits))

        # Iterate until convergence
        delta = 0
        i = random.randint(0, self.N - 1)

        while delta > c_threshold:
            i_prime = random.randint(0, self.N - 1)
            a = self.e_greedy(Q, i, epsilon)
            r = self.reward(i, a)

            delta = alpha * (r + (gamma * np.max(Q[i_prime, :])) - Q[i, a])
            Q[i, a] += delta
            
            i = i_prime
            alpha *= decay
            epsilon *= decay
        return np.argmax(Q, axis=0)
    
    def reward(self, state, action):
        bucket_size = math.ceil(self.features.shape[0] / (action + 1))
        top = (bucket_size ** action) * (self.features.shape[0] - (bucket_size * action))
        bottom = (action + 1) * self.check_common(state, action)
        return top / bottom
    
    def e_greedy(self, Q, i, epsilon):
        if epsilon > random.random():
            return np.argmax(Q[i, :])
        else:
            return random.randint(0, Q.shape[1] - 1)
    

    def check_common(self, feature, split_size):
        values = dict()
        seen = dict()
        for i in range(self.features.shape[0]):
            if i % split_size == 0:
                seen = dict()
            if self.features[i, feature, 1] not in seen:
                if self.features[i, feature, 1] not in values:
                    values[self.features[i, feature, 1]] = 0
                else:
                    values[self.features[i, feature, 1]] += 1
            seen[self.features[i, feature, 1]] = True
        return sum(iter(values.values)) - len(values)