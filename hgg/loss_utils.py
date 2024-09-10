from scipy.interpolate import splprep, splev
import numpy as np
import torch
from tslearn.metrics import SoftDTWLossPyTorch
import torch

# Initialize the Soft-DTW loss function with a small gamma to focus on sharp alignment
soft_dtw_loss = SoftDTWLossPyTorch(gamma=0.05)

def compute_curvature(trajectory):
    """
    Compute the curvature of the trajectory, defined as the angle between consecutive segments.
    The curvature is higher when there are sharper turns.
    """
    diffs = torch.diff(trajectory, dim=0)  # Compute differences between consecutive points
    angles = torch.atan2(diffs[:, 1], diffs[:, 0])  # Compute the angle of each segment
    curvature = torch.abs(torch.diff(angles))  # Absolute difference in angles represents curvature

    return curvature.sum()

def trajectory_similarity_loss(predicted_trajectory, actual_trajectory, alpha=0.5, beta=0.5, segment_length=10, window_step=5):
    """
    Custom loss function for trajectory similarity using segment-wise Soft-DTW from tslearn.
    Incentivizes the model to correctly predict sharp turns.
    """
    total_dtw_loss = 0
    num_segments = 0
    
    # Iterate with a sliding window of length 'segment_length' and step 'window_step'
    for start_idx in range(0, min(len(predicted_trajectory), len(actual_trajectory)) - segment_length + 1, window_step):
        # Define the end of the current segment
        end_idx = start_idx + segment_length
        
        # Extract segments
        pred_segment = predicted_trajectory[start_idx:end_idx]
        true_segment = actual_trajectory[start_idx:end_idx]
        
        # Compute Soft-DTW loss for this segment
        dtw_distance = soft_dtw_loss(pred_segment.unsqueeze(0), true_segment.unsqueeze(0))
        
        # Accumulate the loss
        total_dtw_loss += dtw_distance
        num_segments += 1

    # Calculate the average DTW loss across all overlapping segments
    avg_dtw_loss = total_dtw_loss / num_segments if num_segments > 0 else 0

    # Curvature penalty to incentivize sharp turns
    curvature_penalty = compute_curvature(predicted_trajectory)

    # Combine losses: incentivize sharp turns by subtracting the curvature penalty
    total_loss = alpha * avg_dtw_loss - beta * curvature_penalty

    return total_loss, avg_dtw_loss, curvature_penalty









def euclidean_distance_loss(predicted_trajectory, actual_trajectory):
    """
    Compute the Euclidean distance loss between two trajectories using PyTorch.
    """
    distance = torch.norm(predicted_trajectory - actual_trajectory, dim=1)
    loss = torch.mean(distance)
    return loss
def entropy_gain(current_goals, new_goal, bandwidth=0.1):
    current_goals_tensor = torch.tensor(current_goals, dtype=torch.float32)
    new_goal_tensor = torch.tensor(new_goal, dtype=torch.float32).unsqueeze(0)
    current_entropy = calculate_entropy(current_goals_tensor, bandwidth)
    updated_goals = torch.cat([current_goals_tensor, new_goal_tensor], dim=0)
    updated_entropy = calculate_entropy(updated_goals, bandwidth)
    return updated_entropy - current_entropy		

def calculate_entropy(goals, bandwidth=0.1):
    goals_tensor = torch.tensor(goals, dtype=torch.float32)
    kde = torch.distributions.MultivariateNormal(loc=goals_tensor, covariance_matrix=bandwidth * torch.eye(goals_tensor.size(1)))
    log_density = kde.log_prob(goals_tensor)
    density = torch.exp(log_density)
    entropy = -torch.sum(density * log_density)
    return entropy

def compute_wasserstein_distance(traj1, traj2):
    """
    Computes the Wasserstein distance between two multi-dimensional trajectories using PyTorch.
    
    Parameters:
    traj1 (torch.Tensor): First trajectory, shape (n_points1, n_dims).
    traj2 (torch.Tensor): Second trajectory, shape (n_points2, n_dims).

    Returns:
    torch.Tensor: The Wasserstein distance between the two trajectories.
    """
    # Ensure inputs are of type torch.Tensor
    traj1 = torch.tensor(traj1, dtype=torch.float32) if not isinstance(traj1, torch.Tensor) else traj1
    traj2 = torch.tensor(traj2, dtype=torch.float32) if not isinstance(traj2, torch.Tensor) else traj2

    # Calculate pairwise distance matrix between points in traj1 and traj2
    cost_matrix = torch.cdist(traj1, traj2, p=2)  # Euclidean distance matrix, shape (n_points1, n_points2)

    # Uniform weights for each point (assuming each point has equal weight)
    n_points1, n_points2 = traj1.shape[0], traj2.shape[0]
    weight1 = torch.full((n_points1,), 1.0 / n_points1, dtype=torch.float32)
    weight2 = torch.full((n_points2,), 1.0 / n_points2, dtype=torch.float32)

    # Compute Wasserstein distance using the Kantorovich-Rubinstein duality
    # We multiply the cost matrix by the weight vectors to get the total cost
    wasserstein_distance = torch.sum(cost_matrix * weight1[:, None] * weight2[None, :])

    return wasserstein_distance