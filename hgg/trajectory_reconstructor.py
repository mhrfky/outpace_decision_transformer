import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from networkx import NetworkXNoPath, astar_path
from hgg.utils import calculate_max_distance
from playground2 import time_decorator

class TrajectoryReconstructor:

    def __init__(self, buffer_size=200, n_clusters=200, merge_sample_size=5, merge_when_full=True):
        self.G = nx.Graph()
        self.states = np.array([]).reshape(0, 2)  # Initialize an empty array for states
        self.buffer_size = buffer_size
        self.n_clusters = n_clusters
        self.merge_sample_size = merge_sample_size
        if merge_when_full:
            self.cleaning_func = self.merge_states_using_kmeans
        else:
            self.cleaning_func = self.remove_oldest_states

    def create_graph(self, states, max_distance):
        self.states = states
        for i in range(len(states)):
            self.G.add_node(i, pos=states[i])
            for j in range(i + 1, len(states)):
                distance = np.linalg.norm(states[i] - states[j])
                if distance <= max_distance:
                    self.G.add_edge(i, j, weight=distance)

    @time_decorator
    def add_trajectory(self, trajectory, max_distance):
        start_index = len(self.states)
        self.states = np.vstack((self.states, trajectory))
        for i in range(start_index, len(self.states)):
            self.G.add_node(i, pos=self.states[i])
            for j in range(len(self.states)):
                if i != j:
                    distance = np.linalg.norm(self.states[i] - self.states[j])
                    if distance <= max_distance:
                        self.G.add_edge(i, j, weight=distance)

    @time_decorator
    def shortest_path_trajectory(self, rewards, start_index, end_index):
        try:
            # Define a heuristic based on Euclidean distance
            def heuristic(u, v):
                pos_u = self.G.nodes[u]['pos']
                pos_v = self.G.nodes[v]['pos']
                return np.linalg.norm(np.array(pos_u) - np.array(pos_v))

            shortest_path_indices = astar_path(self.G, source=start_index, target=end_index, heuristic=heuristic, weight='weight')
            shortest_path_states = self.states[shortest_path_indices]
            shortest_path_rewards = rewards[shortest_path_indices]
            return shortest_path_states, shortest_path_rewards
        except NetworkXNoPath:
            print(f"No path found from {start_index} to {end_index}.")
            return None, None

    def concat_original_with_new(self, new_trajectory, len_traj, i, rewards, new_rewards):
        if new_trajectory is None:
            return None, None
        if len_traj == 0:
            return self.states[-100:i], rewards[-100:i]
        return np.vstack((self.states[-100:i], new_trajectory)), np.hstack((rewards[-100:i], new_rewards))

    def get_shortest_path_trajectories_with_yield(self, states, rtgs, top_n, eval_fn=None, pick_n=10, max_distance=None):
        if max_distance is None:
            max_distance = calculate_max_distance(states)

        self.add_trajectory(states, max_distance)
        if len(self.states) >= self.buffer_size:
            self.cleaning_func(max_distance)
        rewards = eval_fn(self.states, None)[0]
        top_indices = np.argsort(rewards)[-top_n:]
        random_indices = np.random.choice(top_indices, size=pick_n, replace=False)
        top_indices = random_indices
        for idx in top_indices:
            for i in range(self.states.shape[0]-100 + 20, self.states.shape[0], 10):
                start_index = i
                new_trajectory, new_rewards = self.shortest_path_trajectory(rewards, start_index, idx)
                if new_trajectory is None or len(new_trajectory) < 1 :
                    continue
                len_traj = len(new_trajectory)

                if  len(new_trajectory) >= 20:  # Check for no path or very short path
                    new_trajectory = new_trajectory[:20]
                    new_rewards = new_rewards[:20]
                    len_traj = len(new_trajectory)

                new_trajectory, new_rtgs = self.concat_original_with_new(new_trajectory, len_traj , i, rewards, new_rewards)
                yield new_trajectory, new_rtgs, len_traj
            

    def get_shortest_jump_tree(self, states, top_n=50, pick_n=10, eval_fn=None):
        max_distance = 1.5 # calculate_max_distance(states)
        state = np.array([0, 0])
        states = np.vstack((state, states))
        self.add_trajectory(states, max_distance)

        if len(self.states) >= self.buffer_size:
            self.cleaning_func(max_distance)

        g_rewards = eval_fn(self.states, None)[0] 
        s_rewards = eval_fn(states, None)[0]
        root = self.find_first_close_state([0,0])
        shortest_path_lengths = nx.single_source_dijkstra_path_length(self.G, root)
        shortest_paths = nx.single_source_dijkstra_path(self.G, root)

        
        
        # Step 2: Sort the paths by the length (weight) in descending order
        sorted_paths = sorted(shortest_paths.items(), key=lambda item: shortest_path_lengths[item[0]], reverse=True)
        paths = np.array([path for _, path in sorted_paths])
        nodes = np.array([node for node, _ in sorted_paths])

        # Step 3: Select the top n paths
        top_n_paths = nodes[:top_n]
        # Select 'pick_n' unique random indices from the range of top_n_paths length
        selected_indices = np.random.choice(len(top_n_paths), size=pick_n, replace=False)

        # If top_n_paths is a list, use list comprehension to pick the paths
        pick_n_paths = top_n_paths[selected_indices]
        
        
        # Step 4: Yield the results
        for  node in pick_n_paths:
            for i in range(1,len(states), 20):
                state = states[i]
                state = self.find_first_close_state(state)
                if state is None:
                    i -= 15
                    continue
                path, temp_rewards = self.shortest_path_trajectory(g_rewards, start_index= state, end_index= node)

                path_to_yield = np.concatenate((states[:i], path[:10]))
                path_rtgs = np.concatenate((s_rewards[:i],temp_rewards[:10]))
                path_rtgs -= path_rtgs[0]
                path_rtgs = path_rtgs[::-1].copy()
                yield path_to_yield, path_rtgs, min(10, len(path))

    @time_decorator
    def merge_states_using_kmeans(self, max_distance):
        """Merge densely clustered states using K-means and keep isolated points to maintain expansiveness."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.states)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        new_states = centroids
        new_graph = nx.Graph()

        # Create new states for clusters and map old indices to new ones, attaching positions as node attributes
        for i in range(len(new_states)):
            new_graph.add_node(i, pos=new_states[i])  # Attach position as node attribute

        # Compute distances between centroids and add edges
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                distance = np.linalg.norm(new_states[i] - new_states[j])

                if i != j and distance <= max_distance:
                    new_graph.add_edge(i, j, weight=distance)  # Use squared distance for consistency
        
        # Update the states and graph
        self.states = new_states
        self.G = new_graph

    @time_decorator
    def remove_oldest_states(self, max_distance):
        """Remove the oldest 100 states and update the graph accordingly."""
        if len(self.states) <= 100:
            return  # Do nothing if there are 100 or fewer states

        # Determine the nodes to remove (the oldest 100)
        nodes_to_remove = list(range(100))

        # Remove the nodes from the graph
        self.G.remove_nodes_from(nodes_to_remove)

        # Update the states array by removing the oldest 100 states
        self.states = self.states[100:]

        # Create a new graph to avoid inconsistencies
        new_graph = nx.Graph()

        # Re-index remaining nodes
        for i, state in enumerate(self.states):
            new_graph.add_node(i, pos=state)

        # Add edges based on max_distance
        for i in range(len(self.states)):
            for j in range(i + 1, len(self.states)):
                distance = np.linalg.norm(self.states[i] - self.states[j])
                if distance <= max_distance:
                    new_graph.add_edge(i, j, weight=distance)

        # Replace the old graph with the new graph
        self.G = new_graph

    def find_first_close_state(self, input_state):
        for i,state in enumerate(self.states):
            distance = np.linalg.norm(np.array(state) - np.array(input_state))
            if distance <= 1:
                return i
        return None