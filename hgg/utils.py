import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class MoveOntheLastPartLoss(nn.Module):
    def __init__(self, threshold):
        super(MoveOntheLastPartLoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss()
        
    def forward(self, achieved_goals, desired_goal):
        epsilon = 1e-10  # Small value to prevent division by zero

        # weight = torch.arange(achieved_goals.shape[1] , 0, step= -1,device = "cuda", dtype = torch.float32)
        weight = torch.arange(1,achieved_goals.shape[1] + 1, device = "cuda", dtype = torch.float32)
        weight = weight / torch.sum(weight)
        
        euclidean_distances = torch.norm(achieved_goals[0] - desired_goal.unsqueeze(0), p=2, dim=1)

        
        below_threshold = (euclidean_distances < self.threshold).float()
        
        dist_denominator = torch.mean(weight * below_threshold * euclidean_distances)
        
        number_of_effected = sum(below_threshold)
        number_of_effected_squared = number_of_effected ** 2
        
        weight_denominator = torch.sum(below_threshold * weight)
        
        # loss = (weight_denominator * number_of_effected_squared - 1)/ (epsilon + dist_denominator)
        loss = euclidean_distances * weight
        return sum(below_threshold)

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3, -2))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3, -2))
        
    def forward(self, x):
        weight_epsilon = Normal(0, 1).sample(self.weight_mu.size()).to(x.device)
        bias_epsilon = Normal(0, 1).sample(self.bias_mu.size()).to(x.device)
        
        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * weight_epsilon
        bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * bias_epsilon
        
        return nn.functional.linear(x, weight, bias)

class BiggerNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[128, 64, 32]):
        super(BiggerNN, self).__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size

        layer_sizes = [input_size] + hidden_layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.output_layer = nn.Linear(layer_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x
    
def generate_random_samples(lower_limit, upper_limit, sample_shape, num_samples):
    """
    Generates a list of NumPy arrays with random samples between the given limits.

    Args:
    - lower_limit: Lower bound of the random samples.
    - upper_limit: Upper bound of the random samples.
    - sample_shape: Shape of each sample (e.g., (2,) for 2D points).
    - num_samples: Number of samples to generate.

    Returns:
    - samples_list: List of NumPy arrays with random samples.
    """
    samples_list = []
    for _ in range(num_samples):
        random_sample = np.random.uniform(lower_limit, upper_limit, sample_shape)
        samples_list.append(random_sample)
    return samples_list
def euclid_distance(p1,p2):
        return np.linalg.norm(p1 - p2)

import torch


def calculate_dynamic_threshold(rtgs, percentile=50):
    """
    Calculate a dynamic threshold based only on the increasing RTG sequence using a specified percentile.
    
    Parameters:
        rtgs (np.array): The sequence of return-to-go values.
        percentile (float): The percentile to use for dynamic threshold calculation.
        
    Returns:
        float: The calculated dynamic threshold.
    """
    # Filter only increasing segments
    window_size = 5
    conv_kernel = np.ones(window_size) / window_size
    smoothed_rtgs = np.convolve(rtgs, conv_kernel, mode='same')
    differences = np.abs(np.diff(smoothed_rtgs))
    threshold = np.percentile(differences, percentile)
    return threshold

def find_diminishing_trajectories(rtgs, length=10, base_threshold=0.01, dynamic_threshold=True, percentile=60):
    """
    Find the starting indices of diminishing trajectories in a sequence of RTGs.
    
    Parameters:
        rtgs (np.array or list): The sequence of return-to-go values.
        length (int): The length of the diminishing trajectory to find.
        base_threshold (float): The base threshold for determining a diminishing trajectory.
        dynamic_threshold (bool): Whether to use a dynamic threshold based on recent RTG values.
        percentile (float): The percentile to use for dynamic threshold calculation.
        
    Returns:
        List of starting indices of diminishing trajectories.
    """
    if isinstance(rtgs, list):
        rtgs = np.array(rtgs)
        
    if dynamic_threshold:
        threshold = calculate_dynamic_threshold(rtgs, percentile=percentile)
    else:
        threshold = base_threshold
    
    diminishing_streaks = []
    n = len(rtgs)
    counter = 0
    
    for i in range(2, n-1):
        if rtgs[i] <= rtgs[i-1] + threshold:
            counter += 1
        else:
            counter = 0

        if counter >= length - 1:
            diminishing_streaks.append((i - length + 1, i - length +11))
            
    return diminishing_streaks



def rescale_array(tensor, old_min, old_max, new_min =-1, new_max = 1):
    rescaled_tensor = (tensor - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return rescaled_tensor

def calculate_max_distance(positions):
    max_distance = 0
    for i in range(len(positions) - 1):
        distance = np.linalg.norm(positions[i+1] - positions[i])
        if distance > max_distance:
            max_distance = distance
    return max_distance

# Function to reorder centroids using DTW
def reorder_centroids_dtw(original_states, centroids):
    _, path = fastdtw(original_states, centroids, dist=euclidean)
    reordered_centroids = [centroids[j] for i, j in path]
    return np.array(reordered_centroids)