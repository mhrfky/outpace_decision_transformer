# import plotly.graph_objects as go
# import numpy as np
# import torch


# def plotly_value_function_graph(self, critic_input_tensor, action, diffusion_goals,
#                                 generate_same_size=True, animation=False):
#     """Generate plotly graph for value function."""
#     if generate_same_size:
#         # Generate data as same size as critic_input_tensor
#         number_of_points = int(np.sqrt(critic_input_tensor.shape[0]))
#         x_debug = torch.linspace(self.min_action[0], self.max_action[0], number_of_points)
#         y_debug = torch.linspace(self.min_action[1], self.max_action[1], number_of_points)
#         X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
#         virtual_diffusion_goals_debug = torch.hstack((X_debug.reshape(-1, 1), Y_debug.reshape(-1, 1))).to(self.device)
#         repeated_state_debug = torch.clone(critic_input_tensor).detach()
#         repeated_state_debug[:, -2:] = virtual_diffusion_goals_debug
#         repeated_state_debug = self.agent.normalize_obs(repeated_state_debug, self.agent.env_name, "diffusion")
#         repeated_action_debug = action.clone()
#         critic_value_debug = self.agent.critic(repeated_state_debug, repeated_action_debug)[0]
#     else:
#         number_of_points = 100
#         x_debug = torch.linspace(self.min_action[0], self.max_action[0], number_of_points)
#         y_debug = torch.linspace(self.min_action[1], self.max_action[1], number_of_points)
#         X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
#         obs_debug = critic_input_tensor[0, :].clone().detach()
#         virtual_diffusion_goals_debug = torch.hstack((X_debug.reshape(-1, 1), Y_debug.reshape(-1, 1))).to(self.device)
#         repeated_state_debug = torch.tile(obs_debug, [len(virtual_diffusion_goals_debug), 1])
#         repeated_state_debug[:, -2:] = virtual_diffusion_goals_debug
#         repeated_state_debug = self.agent.normalize_obs(repeated_state_debug, self.agent.env_name, "diffusion")
#         action_debug = action[0, :].clone().detach()
#         repeated_action_debug = torch.tile(action_debug, [len(virtual_diffusion_goals_debug), 1])
#         critic_value_debug = self.agent.critic(repeated_state_debug, repeated_action_debug)[0]

#     fig = go.Figure()
#     critic_value_surface = np.array(critic_value_debug.detach().cpu()).reshape(X_debug.shape)
#     surface_plot = go.Surface(x=X_debug.cpu().numpy(), y=Y_debug.cpu().numpy(), z=critic_value_surface, colorscale='Viridis')
#     fig.add_trace(surface_plot)
#     fig.update_layout(title="Value Function Visualization", autosize=False, width=1600, height=1400, scene=dict(xaxis=dict(range=[-30, 30]), yaxis=dict(range=[-30, 30]), zaxis=dict(range=[-25, 5])))
#     fig.write_html(f"{self.debug_saved_path}/value_function_{self.counter}.html")

# def plotly_select_env(self):
#     """Select environment frame based on the environment name."""
#     x_coords = np.array([-6, 14, -6, 14, -6, -6, 14, 14, 4, 16, 4, 16, 4, 8, 16, 16, 4, 4, 12, 12, 12, 12]) - 6
#     y_coords = np.array([14, 14, 14, 14, 0, 0, 0, 0, 12, 12, 12, 12, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12]) - 6
#     z_coords = np.array([2, 2, 0, 0, 2, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0])

#     if self.agent.env_name == 'PointUMaze-v0':
#         indices = np.array([0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 9, 11, 1, 3, 18, 16, 16, 14, 14, 12, 12, 8, 8, 9, 9, 22, 11, 21, 11, 11, 10, 10, 13, 13, 15, 15, 17, 17])
#         self.env_frame = go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, i=indices, opacity=0.5, color='red')
#     elif self.agent.env_name == 'PointNMaze-v0':
#         # Similar adjustments for other environment frames
#         pass
#     else:
#         raise NotImplementedError("Environment frame not set for this environment name.")

#     return None

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Sample data points and their values
data_points = np.array([
    (-2, -2, 1), (-2, 4, 2), (-2, 8, 3),
    (0, -2, 4), (0, 4, 5), (0, 8, 6),
    (4, -2, 7), (4, 4, 8), (4, 8, 2),
    (8, -2, 3), (8, 4, 1), (8, 8, 0)
])
x = data_points[:, 0]
y = data_points[:, 1]
values = data_points[:, 2]

# Create a grid to interpolate on
grid_x, grid_y = np.mgrid[-2:8:100j, -2:8:100j]  # 100j indicates 100 points along each dimension

# Interpolate using griddata
grid_values = griddata((x, y), values, (grid_x, grid_y), method='cubic')

# Plotting
plt.figure(figsize=(8, 6))
plt.imshow(grid_values.T, extent=(-2, 8, -2, 8), origin='lower', cmap='viridis')
plt.colorbar(label='Value')
plt.scatter(x, y, c='red', s=50)  # Red dots on the original data points
plt.title('Continuous Gradient Heatmap')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.grid(True)
plt.savefig("a.png")

