import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
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
