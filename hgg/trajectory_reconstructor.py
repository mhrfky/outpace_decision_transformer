import networkx as nx
import numpy as np
from hgg.utils import calculate_max_distance
class TrajectoryReconstructor:
    def __init__(self):
        self.G = nx.Graph()
        self.states = np.array([]).reshape(0,2)  # Initialize an empty array for states

    def create_graph(self, states, max_distance):
        self.states = states
        n = len(self.states)
        self.G = nx.Graph()  # Reset graph and rebuild to maintain consistency
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(self.states[i] - self.states[j])
                if distance <= max_distance:
                    self.G.add_edge(i, j, weight=1)  # Consistent weight calculation

    def add_trajectory(self, trajectory, max_distance):
        start_index = len(self.states)
        self.states = np.vstack((self.states, trajectory))
        n = len(self.states)
        for i in range(start_index, n):
            for j in range(n):
                if i != j:
                    distance = np.linalg.norm(self.states[i] - self.states[j])
                    if distance <= max_distance:
                        self.G.add_edge(i, j, weight=1)  # Ensure connectivity

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

    def shortest_path_trajectory(self, rtgs, end_index):
        shortest_path_indices = nx.dijkstra_path(self.G, source=0, target=end_index, weight='weight')
        shortest_path_states = self.states[shortest_path_indices]
        shortest_path_rtgs = rtgs[shortest_path_indices]
        return shortest_path_states, shortest_path_rtgs

    def get_shortest_path_trajectories_with_yield(self, states, rtgs, top_n, n=10):
        self.reset_graph()
        max_distance = calculate_max_distance(states)
        self.create_graph(states, max_distance)
        top_indices = np.argsort(rtgs)[-top_n:]
        for idx in top_indices:
            len_traj = n
            new_trajectory, new_rtgs = self.shortest_path_trajectory(rtgs, idx)
            if len(new_trajectory) < 3:
                continue
            elif len(new_trajectory) < n + 1:
                len_traj = len(new_trajectory) -1
            
            new_trajectory, new_rtgs = self.concat_original_with_new(new_trajectory, n, rtgs, new_rtgs)
            yield new_trajectory, new_rtgs, len_traj
