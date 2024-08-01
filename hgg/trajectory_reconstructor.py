import numpy as np
import networkx as nx
from hgg.utils import calculate_max_distance
from sklearn.cluster import DBSCAN
from networkx import NetworkXNoPath

class TrajectoryReconstructor:

    def __init__(self, buffer_size = 250, merge_ep = 0.25, merge_sample_size = 5):
        self.G = nx.Graph()
        self.states = np.array([]).reshape(0,2)  # Initialize an empty array for states
        self.buffer_size = buffer_size
        self.merge_ep = merge_ep
        self.merge_sample_size = merge_sample_size
    def create_graph(self, states, max_distance):
        self.states = states

        n = len(self.states)
        self.G = nx.Graph()  # Reset graph and rebuild to maintain consistency
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(self.states[i] - self.states[j])
                if distance <= max_distance:
                    self.G.add_edge(i, j, weight=distance ** 2)  # Consistent weight calculation

    def add_trajectory(self, trajectory, max_distance):
        start_index = len(self.states)
        self.states = np.vstack((self.states, trajectory))
        n = len(self.states)
        for i in range(start_index, n):
            for j in range(n):
                if i != j:
                    distance = np.linalg.norm(self.states[i] - self.states[j])
                    if distance <= max_distance:
                        self.G.add_edge(i, j, weight= distance ** 2)  # Ensure connectivity

    def reset_graph(self):
        self.G.clear()
        self.states = np.array([]).reshape(0, 2)

    def concat_original_with_new(self, new_trajectory, n, rtgs, new_rtgs):
        traj_len = min(n, len(new_trajectory))
        nth_from_end = new_trajectory[-traj_len]
        index = np.where(np.all(self.states == nth_from_end, axis=1))[0][0]
        result_trajectory = np.concatenate((self.states[:index], new_trajectory[-traj_len:]))
        result_rtgs = np.concatenate((rtgs[:index], new_rtgs[-traj_len:]))
        return result_trajectory, result_rtgs

    def shortest_path_trajectory(self, rewards, end_index):
        try:
            shortest_path_indices = nx.dijkstra_path(self.G, source=0, target=end_index, weight='weight')
            shortest_path_states = self.states[shortest_path_indices]
            shortest_path_rtgs = rewards[shortest_path_indices]
            return shortest_path_states, shortest_path_rtgs
        except NetworkXNoPath:
            print(f"No path found to index {end_index}.")
            return None, None

    def get_shortest_path_trajectories_with_yield( self, states, rtgs, top_n, eval_fn = None, n=10, max_distance = None):
        if max_distance is None:
            max_distance = calculate_max_distance(states)

        self.add_trajectory(states, max_distance)
        rewards =  eval_fn(self.states, None)[0] 
        top_indices = np.argsort(rewards)[-top_n:]
        for idx in top_indices:
            new_trajectory, new_rewards = self.shortest_path_trajectory(rewards, idx)
            if new_trajectory is None or len(new_trajectory) < 3:  # Check for no path or very short path
                continue
            len_traj = min(n, len(new_trajectory) - 1) if len(new_trajectory) > n else len(new_trajectory) - 1
            # new_trajectory, new_rtgs = self.concat_original_with_new(new_trajectory, len_traj, rtgs, new_rtgs)
            new_rtgs = new_rewards - new_rewards[-1]
            if new_trajectory is not None:
                 yield new_trajectory, new_rtgs, len_traj
        if len(self.states) > self.buffer_size: 
            self.merge_states_using_dbscan()

    def merge_states_using_dbscan(self):
        """Merge densely clustered states using DBSCAN and keep isolated points to maintain expansiveness."""
        clustering = DBSCAN(eps=self.merge_ep, min_samples=self.merge_sample_size).fit(self.states)
        labels = clustering.labels_
        new_states = []
        new_graph = nx.Graph()

        # Create new states for clusters and map old indices to new ones
        new_index_map = {}
        cluster_seen = set()
        for idx, label in enumerate(labels):
            if label == -1:  # Handle noise by treating it as its own cluster
                centroid = self.states[idx]
                new_states.append(centroid)
                new_node_index = len(new_states) - 1
                new_index_map[idx] = new_node_index
            elif label not in cluster_seen:  # Process new cluster
                cluster_seen.add(label)
                cluster_indices = np.where(labels == label)[0]
                centroid = np.mean(self.states[cluster_indices], axis=0)
                new_states.append(centroid)
                new_node_index = len(new_states) - 1
                for cluster_idx in cluster_indices:
                    new_index_map[cluster_idx] = new_node_index

        # Reconnect nodes in the new graph
        for idx, new_idx in new_index_map.items():
            connected_indices = list(self.G.adj[idx])
            for connected_idx in connected_indices:
                if connected_idx in new_index_map:
                    new_connected_idx = new_index_map[connected_idx]
                    if new_idx != new_connected_idx:  # Prevent self-loops
                        # Aggregate or average existing weights
                        weight = self.G.edges[idx, connected_idx]['weight']
                        if new_graph.has_edge(new_idx, new_connected_idx):
                            new_graph[new_idx][new_connected_idx]['weight'] = \
                                (new_graph[new_idx][new_connected_idx]['weight'] + weight) / 2
                        else:
                            new_graph.add_edge(new_idx, new_connected_idx, weight=weight)

        self.states = np.array(new_states)
        self.G = new_graph
