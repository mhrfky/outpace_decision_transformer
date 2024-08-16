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

