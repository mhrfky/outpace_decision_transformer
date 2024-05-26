import torch
import torch.nn.functional as F
import torch.nn as nn


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
        return sum(loss)
    