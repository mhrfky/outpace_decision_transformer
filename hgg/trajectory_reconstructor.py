import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from networkx import NetworkXNoPath, astar_path
from hgg.utils import calculate_max_distance
from debug_utils import time_decorator
class TrajectoryReconstructor:

    def __init__(self, buffer_size=200, n_clusters=200, merge_sample_size=5, merge_when_full=True, final_goal=[20,20], max_step=256):
        self.G = nx.Graph()
        self.states = np.array([]).reshape(0, 2)  # Initialize an empty array for states
        self.timesteps = np.array([])  # Initialize an empty array for timesteps
        self.buffer_size = buffer_size
        self.n_clusters = n_clusters
        self.merge_sample_size = merge_sample_size
        self.final_goal = final_goal
        self.max_step = max_step
        self.time_predictor = TimePredictor(2, evaluator=self.weighted_mean_timestep, limits=[[-2,20],[-2,20]])
        if merge_when_full:
            self.cleaning_func = self.merge_states_using_kmeans
        else:
            self.cleaning_func = self.remove_oldest_states

    def create_graph(self, states, max_distance):
        self.states = states
        for i in range(len(states)):
            self.G.add_node(i, pos=states[i], timestep=i)  # Add timestep as a node attribute
            for j in range(i + 1, len(states)):
                distance = np.linalg.norm(states[i] - states[j])
                if distance <= max_distance:
                    self.G.add_edge(i, j, weight=distance)

    @time_decorator
    def add_trajectory(self, trajectory, max_distance):
        start_index = len(self.states)
        timesteps = np.arange(start_index, start_index + len(trajectory))  # Generate timesteps for new trajectory
        self.states = np.vstack((self.states, trajectory))
        self.timesteps = np.hstack((self.timesteps, timesteps))  # Append new timesteps

        for i in range(start_index, len(self.states)):
            self.G.add_node(i, pos=self.states[i], timestep=self.timesteps[i])
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
            # print(f"No path found from {start_index} to {end_index}.")
            return None, None
    def get_shortest_jump_tree(self, states, top_n=50, pick_n=10, eval_fn=None):
            max_distance = 1.5 # calculate_max_distance(states)
            start_state = np.array([0, 0])
            
            states = np.vstack((start_state, states))
            self.add_trajectory(states, max_distance)
            if len(self.states) >= self.buffer_size:
                self.cleaning_func(max_distance)
            self.time_predictor.train(epochs=100)
            g_rewards = eval_fn(self.states, None)[0] 
            s_rewards = eval_fn(states, None)[0]
            root = self.find_first_close_state([0,0])
            shortest_path_lengths = nx.single_source_dijkstra_path_length(self.G, root)
            shortest_paths = nx.single_source_dijkstra_path(self.G, root)

            
            
            # Step 2: Sort the paths by the length (weight) in descending order
            sorted_paths = sorted(shortest_paths.items(), key=lambda item: shortest_path_lengths[item[0]], reverse=True)
            paths = np.array([path for _, path in sorted_paths], dtype=object)
            nodes_ids = np.array([node for node, _ in sorted_paths])

            # Step 3: Select the top n paths
            top_n_paths = nodes_ids[:top_n]
            # Select 'pick_n' unique random indices from the range of top_n_paths length
            nodes_t = torch.tensor(self.states[top_n_paths], device='cuda', dtype=torch.float32)
            timesteps_t = self.time_predictor.predict(nodes_t)
            timesteps = timesteps_t.cpu().detach().numpy().flatten()
            top_n_paths = top_n_paths[timesteps.argsort()]
            pick_n_paths = top_n_paths[-pick_n:]
            # If top_n_paths is a list, use list comprehension to pick the paths
            
            
            # Step 4: Yield the results
            for  node in pick_n_paths:
                for i in range(1,len(states), 20):
                    start_state = states[i]
                    start_state = self.find_first_close_state(start_state)
                    if start_state is None:
                        i -= 15
                        continue
                    path, temp_rewards = self.shortest_path_trajectory(g_rewards, start_index= start_state, end_index= node)

                    path_to_yield = np.concatenate((states[:i], path[:20]))
                    path_rtgs = np.concatenate((s_rewards[:i],temp_rewards[:20]))
                    path_rtgs -= path_rtgs[0]
                    path_rtgs = path_rtgs[::-1].copy()
                    yield path_to_yield, path_rtgs, min(20, len(path))

    @time_decorator
    def merge_states_using_kmeans(self, max_distance):
        """Merge densely clustered states using K-means and keep isolated points to maintain expansiveness."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.states)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        min_timesteps = np.full(self.n_clusters, np.inf)

        # Update min_timesteps for each centroid
        for i, label in enumerate(labels):
            min_timesteps[label] = min(min_timesteps[label], self.timesteps[i])

        new_states = centroids
        new_timesteps = min_timesteps
        new_graph = nx.Graph()

        # Create new nodes for centroids and attach position and timestep as node attributes
        for i in range(len(new_states)):
            new_graph.add_node(i, pos=new_states[i], timestep=new_timesteps[i])  # Attach position and timestep as node attributes

        # Compute distances between centroids and add edges
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                distance = np.linalg.norm(new_states[i] - new_states[j])

                if i != j and distance <= max_distance:
                    new_graph.add_edge(i, j, weight=distance)

        # Update the states, timesteps, and graph
        self.states = new_states
        self.timesteps = new_timesteps
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

        # Update the states and timesteps arrays by removing the oldest 100 states
        self.states = self.states[100:]
        self.timesteps = self.timesteps[100:]

        # Create a new graph to avoid inconsistencies
        new_graph = nx.Graph()

        # Re-index remaining nodes
        for i, state in enumerate(self.states):
            new_graph.add_node(i, pos=state, timestep=self.timesteps[i])

        # Add edges based on max_distance
        for i in range(len(self.states)):
            for j in range(i + 1, len(self.states)):
                distance = np.linalg.norm(self.states[i] - self.states[j])
                if distance <= max_distance:
                    new_graph.add_edge(i, j, weight=distance)

        # Replace the old graph with the new graph
        self.G = new_graph

    def find_first_close_state(self, input_state):
        for i, state in enumerate(self.states):
            distance = np.linalg.norm(np.array(state) - np.array(input_state))
            if distance <= 1:
                return i
        return None

    def weighted_mean_timestep(self, input_state, d=1):

        distances = np.linalg.norm(self.states - input_state, axis=1)
        within_distance_indices = np.where(distances <= d)[0]

        if len(within_distance_indices) == 0:
            return 1  # No states within distance d

        # Calculate weights as the inverse of squared distances
        squared_distances = distances[within_distance_indices] ** 2
        weights = 1 / squared_distances

        # Calculate the weighted mean of timesteps
        nearby_timesteps = self.timesteps[within_distance_indices]
        weighted_mean = np.sum(weights * nearby_timesteps) / np.sum(weights)
        weighted_mean /= self.max_step
        return weighted_mean
import torch
import torch.nn as nn
import torch.optim as optim
class TimePredictor:
    def __init__(self, input_size, hidden_size=64, output_size=1, learning_rate=0.001, evaluator=None, limits=[[-2,10],[-2,10]]):
        self.model = TimePredictionModel(input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.evaluator = evaluator
        self.limits = limits
    def get_batch(self,batch_size):
        xs = []
        ys = []
        for _ in range(batch_size):
            x = [np.random.uniform(self.limits[0][0], self.limits[0][1]), np.random.uniform(self.limits[1][0], self.limits[1][1])]
            y = self.evaluator(x)
            x = torch.tensor(x, device='cuda', dtype=torch.float32)
            y = torch.tensor(y, device='cuda', dtype=torch.float32)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)
    def train(self,  epochs=100):
        xs,ys = self.get_batch(1000)

        for _ in range(len(xs)):
            self.optimizer.zero_grad()
            output = self.model(xs)
            loss = self.criterion(output, ys)
            loss.backward()
            self.optimizer.step()
            
    def predict(self, X):
        return self.model(X)    
    
class TimePredictionModel(nn.Module):
    def __init__(self, input_size):
        super(TimePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64).cuda()
        self.fc2 = nn.Linear(64, 32).cuda()
        self.fc3 = nn.Linear(32, 1).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    