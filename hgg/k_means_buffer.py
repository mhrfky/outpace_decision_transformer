import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class StateBuffer:
    def __init__(self, max_size, n_clusters):
        self.max_size = max_size
        self.n_clusters = n_clusters
        self.states = []
        self.counts = []
        self.total_counts_sum = 0

    def add_state(self, state):
        if len(self.states) < self.max_size:
            self.states.append(state)
            self.counts.append(1)
            self.total_counts_sum += 1
        else:
            self.merge_states(state)

    def merge_states(self, new_state):
        # Append the new state and its count
        self.states.append(new_state)
        self.counts.append(1)
        self.total_counts_sum += 1
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        clusters = kmeans.fit_predict(self.states)
        
        # Create new states and counts based on centroids
        new_states = []
        new_counts = []
        for i in range(self.n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            cluster_states = np.array([self.states[j] for j in cluster_indices])
            cluster_counts = np.array([self.counts[j] for j in cluster_indices])
            
            # Calculate the weighted centroid
            weighted_centroid = np.average(cluster_states, axis=0, weights=cluster_counts)
            new_states.append(weighted_centroid)
            new_counts.append(np.sum(cluster_counts))
        
        # Update the buffer with the merged states and counts
        self.states = new_states
        self.counts = new_counts

    def get_states(self):
        return self.states, self.counts
    
    def get_close_state_count(self, state, threshold=0.1):
        """
        Get the number of states that are within a certain distance threshold from the given state.
        """
        distances = cdist([state], self.states)[0]
        close_state_count = np.sum(distances < threshold)
        return close_state_count
    
    def get_close_state_counts_sum(self, state, threshold=0.1):
        """
        Get the sum of the counts of states that are within a certain distance threshold from the given state.
        """
        distances = cdist([state], self.states)[0]
        close_state_indices = np.where(distances < threshold)[0]
        close_state_counts_sum = np.sum([self.counts[i] for i in close_state_indices])
        return close_state_counts_sum