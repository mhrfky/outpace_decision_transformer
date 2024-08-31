from scipy.interpolate import splprep, splev
import numpy as np
import torch
from tslearn.metrics import SoftDTWLossPyTorch


soft_dtw_loss = SoftDTWLossPyTorch(gamma=0.1)


def cubic_spline_interpolation_loss(trajectory_t , residuals_t):
    trajectory_np = trajectory_t.squeeze(0).detach().cpu().numpy()
    residuals_t = residuals_t.squeeze(0)
    x = trajectory_np[:,0]
    y = trajectory_np[:,1]
    if len(x) < 3:
        return torch.tensor(0, device = 'cuda', dtype=torch.float32)
    tck,u = splprep([x,y],s=1, k=2)

    u_extrapolated = np.linspace(0, 1.5, num=len(x) + len(residuals_t))  # Extend beyond the original data range
    x_extrapolated, y_extrapolated = splev(u_extrapolated, tck)
    spline_t = torch.tensor(np.vstack((x_extrapolated, y_extrapolated)).T, device = 'cuda' ,dtype=torch.float32)
    # plot_two_array_positions(trajectory_np,np.vstack((x_extrapolated, y_extrapolated)).T )
    _, dtw_distance, _, _ = trajectory_similarity_loss(spline_t[-5:], residuals_t)
    return dtw_distance

def dtw_loss(predicted_trajectory, actual_trajectory):
    """
    Compute the Dynamic Time Warping (DTW) distance between two trajectories using PyTorch.
    """
    n, m = predicted_trajectory.size(0), actual_trajectory.size(0)
    dtw_matrix = torch.full((n + 1, m + 1), float('inf'), device=predicted_trajectory.device)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = torch.norm(predicted_trajectory[i - 1] - actual_trajectory[j - 1])
            min_val = torch.min(
                torch.min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1]), 
                dtw_matrix[i - 1, j - 1]
            )
            dtw_matrix[i, j] = cost + min_val

    return dtw_matrix[n, m]

def trajectory_similarity_loss(predicted_trajectory, actual_trajectory, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Custom loss function for trajectory similarity.
    """
    # Compute the Soft-DTW loss
    dtw_distance = soft_dtw_loss(predicted_trajectory.unsqueeze(0), actual_trajectory.unsqueeze(0)).mean()



    # Smoothness regularization (L2 norm of differences between consecutive points)
    smoothness_reg = torch.sum(torch.norm(torch.diff(predicted_trajectory, dim=0), dim=1)**2)

    # Combine losses
    total_loss = alpha * dtw_distance  + gamma * smoothness_reg

    return total_loss, dtw_distance, smoothness_reg
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