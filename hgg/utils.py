import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

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
        
        self.weight_epsilon = None
        self.bias_epsilon = None

    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        self.weight_epsilon = Normal(0, 1).sample(self.weight_mu.size()).to(x.device)
        self.bias_epsilon = Normal(0, 1).sample(self.bias_mu.size()).to(x.device)
        
        weight = self.weight_mu + weight_sigma * self.weight_epsilon
        bias = self.bias_mu + bias_sigma * self.bias_epsilon
        
        return nn.functional.linear(x, weight, bias)

class BayesianNN(nn.Module):
    def __init__(self):
        super(BayesianNN, self).__init__()
        self.fc1 = BayesianLinear(2, 128)
        self.fc2 = BayesianLinear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

