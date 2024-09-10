import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.stats import entropy
from hgg.utils import calculate_max_distance
class KMeansRegulatedSubtrajBuffer:
    def __init__(self, max_size=100, n_centroids=200, extract_top_n=5, val_eval_fn=None, path_evaluator = None, final_goal = [4,0]):
        self.max_size = max_size
        self.n_centroids = n_centroids
        self.extract_top_n = extract_top_n
        self.trajectory_buffer = []  # Buffer for sub-trajectories
        self.trajectory_len_buffer = []
        self.centroid_buffer = np.zeros([n_centroids,2])  # Buffer for centroids
        self.val_eval_fn = val_eval_fn
        self.path_evaluator  = path_evaluator
        self.final_goal = final_goal
    def add_trajectory(self, trajectory):
        # Merge the raw trajectory into centroids
        self._merge_to_centroids(trajectory)
        
        # Extract sub-trajectories from the merged centroids
        subtrajectories, subtrajectory_lens = self.extract_subtrajectories(trajectory)
        
        # Add sub-trajectories to the trajectory buffer
        for subtraj, subtraj_len in zip(subtrajectories, subtrajectory_lens):
            if len(self.trajectory_buffer) >= self.max_size:
                self.trajectory_buffer.pop(0)  # Remove the oldest trajectory if the buffer is full
                self.trajectory_len_buffer.pop(0)
            self.trajectory_buffer.append(subtraj)
            self.trajectory_len_buffer.append(subtraj_len)

    def _merge_to_centroids(self, trajectory):
        # Convert trajectory to numpy array if needed
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)

        # Handle the case where the centroid buffer is empty
        if len(self.centroid_buffer) == 0:
            all_states = trajectory
        else:
            # Flatten the centroids and trajectory for k-means clustering
            all_states = np.vstack([np.vstack(self.centroid_buffer), trajectory])

        # Perform k-means clustering with a consistent number of centroids
        kmeans = KMeans(n_clusters=self.n_centroids, random_state=0).fit(all_states)
        new_centroids = kmeans.cluster_centers_

        # Update centroid buffer with the new centroids
        self.centroid_buffer = new_centroids


    def extract_subtrajectories(self, trajectory):
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
        
        # Ensure calculate_max_distance is defined and returns a sensible value
        max_distance = calculate_max_distance(trajectory)
        
        # Create a graph where nodes are trajectory points and edges are weighted by distance
        def create_graph(traj, max_dist):
            G = nx.Graph()
            for i in range(len(traj)):
                for j in range(i + 1, len(traj)):
                    distance = np.linalg.norm(traj[i] - traj[j])
                    if distance <= max_dist:
                        G.add_edge(i, j, weight=distance)
            return G
        
        G = create_graph(trajectory, max_distance)
        
        # Compute shortest paths and their lengths from node 0
        shortest_path_lengths_in_ids,shortest_paths_in_ids = nx.single_source_dijkstra(G, 0)

        
        if self.final_goal is not None:
            # Add the final goal to the graph
            G.add_node(len(trajectory), pos=self.final_goal)

            # Attempt to connect the final goal to the nearest nodes
            final_goal_connected = False
            for i, traj_point in enumerate(trajectory):
                distance = np.linalg.norm(self.final_goal - traj_point)
                if distance <= 2:
                    G.add_edge(i, len(trajectory), weight=0)
                    final_goal_connected = True

            if final_goal_connected:
                # Compute shortest paths and their lengths from the final goal
                final_shortest_path_lengths, _ = nx.single_source_dijkstra(G, len(trajectory))

                # Compute path differences
                path_diff = {k: shortest_path_lengths_in_ids.get(k, 0) - final_shortest_path_lengths.get(k, 0) * 5
                             for k in shortest_path_lengths_in_ids.keys()}
            else:
                # Remove the final goal node if it is not connected
                G.remove_node(len(trajectory))
                # Subtract 0 from all path lengths since none are connected
                path_diff = {k: shortest_path_lengths_in_ids.get(k, 0) - 0 for k in shortest_path_lengths_in_ids.keys()}
        else:
            path_diff = shortest_path_lengths_in_ids

        # Sort paths by their length (longer paths first)
        sorted_paths_in_ids = sorted(shortest_paths_in_ids.items(), key=lambda item: path_diff[item[0]], reverse=True)
        
        # Ensure we do not exceed the number of available paths
        num_paths_to_extract = min(self.extract_top_n, len(sorted_paths_in_ids))
        
        # Extract the top N longest paths
        top_n_paths_nodes = [path for _, path in sorted_paths_in_ids[:num_paths_to_extract]]
        top_n_paths_lengths = [path_diff[path[-1]] for path in top_n_paths_nodes]
        
        subtrajectories = [trajectory[path] for path in top_n_paths_nodes]
        
        return subtrajectories, top_n_paths_lengths


    

    def sample(self):

        # Randomly select 30 new sub-trajectories from the buffer
        indices = np.argsort(self.trajectory_len_buffer)[::-1][:30]        
        
        subtrajectories = np.array(self.trajectory_buffer, dtype=object)[indices].tolist()
        

        
        # Calculate entropy gains for these combined sub-trajectories
        gains = np.array([ self.path_evaluator(subtraj[-1]) for subtraj in subtrajectories]).flatten()
        
        # Sort by entropy gain and select the best 10
        final_best_10 = np.argsort(gains)[:10]
        result_subtrajectories = [subtrajectories[i] for i in final_best_10]
        
        # Yield the final best 10 sub-trajectories
        for subtrajectory in result_subtrajectories:
            if len(subtrajectory) == 1:
                subtrajectory = np.array([subtrajectory[0], subtrajectory[0]])
            rewards = self.val_eval_fn(subtrajectory, None)[0]
            rewards -= rewards[0]
            rtgs = rewards[::-1].copy()
            yield subtrajectory, rtgs, len(subtrajectory) - 1
            
    def check_if_close(self, input_state, d=1):
            
            distances = np.linalg.norm(self.centroid_buffer - input_state, axis=1)
            within_distance_indices = np.where(distances <= d)[0]
            return int(len(within_distance_indices) > 0)
    def get_closest_points(self,sampled_goals):
        closest_points = []

        # Loop through each sampled goal
        for goal in sampled_goals:
            # Calculate the Euclidean distance from the goal to each point in the centroid buffer
            distances = np.linalg.norm(self.centroid_buffer - goal, axis=1)
            
            # Find the index of the closest point
            closest_index = np.argmin(distances)
            
            # Append the closest point to the result list
            closest_points.append(self.centroid_buffer[closest_index])

        return closest_points