import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from networkx import NetworkXNoPath, astar_path
from hgg.utils import calculate_max_distance
from debug_utils import time_decorator
class TrajectoryBufferer:
    def __init__(self, buffer_size=100, val_eval_fn=None, max_len=30):
        self.buffer_size = buffer_size
        self.extract_top_n = 5
        self.subtrajectory_buffer = []
        self.entropies = []  # To store entropy values of subtrajectories
        self.len_of_trajectory = []
        self.val_eval_fn = val_eval_fn
        self.trajectory_length = []

    def add_trajectory(self, trajectory):
        subtrajectories = self.extract_subtrajectories(trajectory)
        self.add_to_buffer(subtrajectories)

    def add_to_buffer(self, subtrajectories):
        for subtrajectory in subtrajectories:
            entropy = self.calculate_entropy(subtrajectory)
            self.subtrajectory_buffer.append(subtrajectory)
            self.entropies.append(entropy)
        
        # Trim buffer if it exceeds the buffer size
        if len(self.subtrajectory_buffer) > self.buffer_size:
            excess = len(self.subtrajectory_buffer) - self.buffer_size
            self.subtrajectory_buffer = self.subtrajectory_buffer[excess:]
            self.entropies = self.entropies[excess:]

    def calculate_entropy(self, subtrajectory):
        # Calculate variance as a proxy for entropy
        variance = np.var(subtrajectory, axis=0)
        entropy = np.sum(variance)
        return entropy

    def sample(self):
        # Sort subtrajectories by stored entropy values in descending order
        sorted_indices = np.argsort(self.entropies)[::-1]
        sorted_subtrajectories = [self.subtrajectory_buffer[i] for i in sorted_indices]
        
        # Sample top 30 (or less) subtrajectories with the highest entropy
        sampled_subtrajectories = sorted_subtrajectories[:min(len(sorted_subtrajectories), 30)]
        
        for subtrajectory in sampled_subtrajectories:
            rewards = self.val_eval_fn(subtrajectory, None)[0]
            rewards -= rewards[0]
            rtgs = rewards[::-1].copy()
            yield subtrajectory, rtgs, min(len(subtrajectory) - 1, 30)

    def extract_subtrajectories(self, trajectory):
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
        max_distance = calculate_max_distance(trajectory)
        G = self.create_graph(trajectory, max_distance)
        shortest_paths_in_ids = nx.single_source_dijkstra_path(G, 0)
        shortest_path_lengths_in_ids = nx.single_source_dijkstra_path_length(G, 0)

        sorted_paths_in_ids = sorted(shortest_paths_in_ids.items(), key=lambda item: shortest_path_lengths_in_ids[item[0]], reverse=True)
        paths = np.array([path for _, path in sorted_paths_in_ids], dtype=object)

        top_n_paths = paths[:self.extract_top_n]

        subtrajectories = []
        for path in top_n_paths:
            subtrajectories.append(trajectory[path])

        return subtrajectories
                                                              
    def create_graph(self, states, max_distance):
        G = nx.Graph()
        for i in range(len(states)):
            G.add_node(i, pos=states[i], timestep=i)  # Add timestep as a node attribute
            for j in range(i + 1, len(states)):
                distance = np.linalg.norm(states[i] - states[j])
                if distance <= max_distance:
                    G.add_edge(i, j, weight=distance ** 2)
        return G
