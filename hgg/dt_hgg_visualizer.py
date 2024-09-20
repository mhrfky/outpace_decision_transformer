import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import itertools
from hgg.value_estimator import ValueEstimator
import torch
import matplotlib.image as mpimg
import os
import cv2  # For perspective transformation

class Visualizer:
    def __init__(self, dt_sampler):
        self.dt_sampler = dt_sampler
        self.value_estimator: ValueEstimator = dt_sampler.value_estimator
        self.limits = self.dt_sampler.limits
        self.history_of_number_of_states_in_reconstructor = np.array([]).reshape(0, 1)

        # Define the base path for background images
        base_image_path = os.path.join(os.getcwd(), '../../../../hgg')

        # Define the background images and perspective transformation matrices for each maze type
        self.env_settings = {
            "PointSpiralMaze-v0": {
                "background_image": os.path.join(base_image_path, "PointSpiralMaze_background.png"),
                "world_coordinate": np.float32([[8, -8], [-8, -8], [-8, 8], [8, 8]]),
                "image_coordinate": np.float32([[395, 395], [85, 395], [85, 85], [395, 85]])
            },
            "PointUMaze-v0": {
                "background_image": os.path.join(base_image_path, "PointUMaze_background.png"),
                "world_coordinate": np.float32([[8, 8], [0, 8], [0, 0], [8, 0]]),
                "image_coordinate": np.float32([[355, 127], [124, 123], [124, 357], [355, 355]])
            },
            "PointNMaze-v0": {
                "background_image": os.path.join(base_image_path, "PointNMaze_background.png"),
                "world_coordinate": np.float32([[8, 16], [0, 16], [0, 0], [8, 0]]),
                "image_coordinate": np.float32([[317, 85], [162, 85], [162, 395], [316, 394]])
            },
            "AntMazeSmall-v0": {
                "background_image": os.path.join(base_image_path, "PointUMaze_background.png"),
                "world_coordinate": np.float32([[8, 8], [0, 8], [0, 0], [8, 0]]),
                "image_coordinate": np.float32([[355, 127], [124, 123], [124, 357], [355, 355]])
            },
        }

        # Check if the maze environment exists in the settings
        env_name = dt_sampler.env_name
        if env_name in self.env_settings:
            # Retrieve the background image and transformation matrix if environment exists
            settings = self.env_settings[env_name]
            self.background_image = mpimg.imread(settings["background_image"])
            self.M = cv2.getPerspectiveTransform(settings["world_coordinate"], settings["image_coordinate"])
            self.use_projection = True  # Flag to indicate that projection and background should be used
        else:
            # No background or projection for environments not in the dictionary
            self.background_image = None
            self.M = None
            self.use_projection = False

    def calculate_perspective_transform(self, world_position):
        # If projection is not used, return the original world position
        if not self.use_projection:
            return world_position

        # Ensure the input is a 2D point
        if isinstance(world_position, (float, np.float32, np.float64, int)):
            raise TypeError(f"Expected a 2D point, but got a scalar: {world_position}")
        if len(world_position) != 2:
            raise ValueError(f"Expected a 2D point, but got: {world_position}")

        # Convert to the correct shape for perspective transformation
        world_position = np.array(world_position, dtype=np.float32).reshape(-1, 1, 2)
        pixel_position = cv2.perspectiveTransform(world_position, self.M)
        pixel_x, pixel_y = pixel_position[0][0]
        return int(pixel_x), int(pixel_y)

    def visualize_value_heatmaps_for_debug(self, goals_predicted_during_training, predicted_states, predicted_rtgs):
        dt_sampler = self.dt_sampler
        value_estimator = self.value_estimator
        num_points_per_axis = 10
        
        # Get the combined grid points (x, y[, z])
        combined_heatmap = self.create_combined_np(num_points_per_axis)
        
        if combined_heatmap.shape[1] == 3:  # If 3D
            # Get the total, achieved, exploration, and q-values for all grid points
            total_values, achieved_values, exploration_values, q_values = value_estimator.get_state_values(combined_heatmap, None)
            
            # Reshape the values to a 3D grid (num_points_per_axis x num_points_per_axis x num_points_per_axis)
            achieved_values_reshaped = achieved_values.reshape(num_points_per_axis, num_points_per_axis, num_points_per_axis)
            q_values_reshaped = q_values.reshape(num_points_per_axis, num_points_per_axis, num_points_per_axis)
            combined_heatmap = combined_heatmap.reshape(num_points_per_axis, num_points_per_axis, num_points_per_axis, 3)

            # Average over the z-dimension to get 2D representations (num_points_per_axis x num_points_per_axis)
            achieved_values_2d = np.mean(achieved_values_reshaped, axis=2).reshape(num_points_per_axis**2)
            q_values_2d = np.mean(q_values_reshaped, axis=2).reshape(num_points_per_axis**2)
            combined_heatmap = np.mean(combined_heatmap, axis=2).reshape(num_points_per_axis**2, 3)
            combined_heatmap_2d = combined_heatmap[:, :2]
            # Reshape combined_heatmap to 2D (only x, y dimensions) to match the shape of the 2D values
            # combined_heatmap_2d = combined_heatmap[:, :2].reshape(num_points_per_axis**2, 2)
        else:
            # For 2D case, no averaging needed
            total_values, achieved_values, exploration_values, q_values = value_estimator.get_state_values(combined_heatmap, None)
            achieved_values_2d = achieved_values
            q_values_2d = q_values
            
            # Keep the combined_heatmap as it is (x, y)
            combined_heatmap_2d = combined_heatmap[:, :2]

        # Plotting each figure separately and saving
        self.plot_and_save(combined_heatmap_2d, q_values_2d, 'Q Heatmap', dt_sampler.episode)
        self.plot_and_save(combined_heatmap_2d, achieved_values_2d, 'Aim Heatmap', dt_sampler.episode)

        self.visualize_trajectories_on_time(dt_sampler.episode)
        self.visualize_sampled_goals(dt_sampler.episode)
        self.visualize_sampled_trajectories(dt_sampler.episode)
        self.visualize_predicted_states(predicted_states, dt_sampler.episode)

        # Predict time step using the Bayesian predictor and plot
        predictor = self.dt_sampler.bayesian_predictor
        combined_heatmap_t = torch.tensor(combined_heatmap, device="cuda", dtype=torch.float32)
        t = predictor.predict(combined_heatmap_t)
        timestep_predictions = np.hstack((combined_heatmap[:, :2], t.reshape(-1, 1)))
        self.plot_and_save(timestep_predictions[:, :2], timestep_predictions[:, 2], 'Time Step Prediction', dt_sampler.episode)

    def plot_and_save(self, points, values, title, episode):
        plt.figure(figsize=(10, 8))

        # Extract x and y coordinates
        x, y = points[:, 0], points[:, 1]

        # Create a grid for interpolation
        grid_x, grid_y = np.mgrid[self.limits[0][0]:self.limits[0][1]:100j, 
                                self.limits[1][0]:self.limits[1][1]:100j]
        
        # Interpolate scattered points onto the grid
        grid_values = griddata((x, y), values, (grid_x, grid_y), method='cubic')

        # Plot the interpolated grid values as a heatmap
        plt.imshow(grid_values.T, extent=(self.limits[0][0], self.limits[0][1], 
                                        self.limits[1][0], self.limits[1][1]), 
                origin='lower', cmap='viridis', aspect='auto')
        
        plt.colorbar(label='Value')
        plt.grid(True)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(title)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_{title}.jpg', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_sampled_trajectories(self, episode):
        plt.figure()

        dt_sampler = self.dt_sampler
        trajectories = dt_sampler.debug_trajectories
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'cyan', 'magenta', 'brown']

        # Check if background image is available and projection is used
        if self.use_projection and self.background_image is not None:
            plt.imshow(self.background_image, extent=(0, self.background_image.shape[1], self.background_image.shape[0], 0))
            for i, traj in enumerate(trajectories):
                traj = np.array([self.calculate_perspective_transform(p) for p in traj])
                plt.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)], linewidth=1)
                plt.scatter(traj[:, 0], traj[:, 1], color="grey", edgecolors='k')
                plt.scatter(traj[-1, 0], traj[-1, 1], color='red', edgecolor='k', s=50)
            plt.xlim(0, self.background_image.shape[1])
            plt.ylim(self.background_image.shape[0], 0)
            plt.axis('off')
        else:
            for i, traj in enumerate(trajectories):
                traj = np.array(traj)
                plt.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)], linewidth=1)
                plt.scatter(traj[:, 0], traj[:, 1], color="grey", edgecolors='k')
                plt.scatter(traj[-1, 0], traj[-1, 1], color='red', edgecolor='k', s=50)
            plt.grid(True)
            plt.axis('on')
            plt.xlim(self.limits[0][0], self.limits[0][1])
            plt.ylim(self.limits[1][0], self.limits[1][1])

        plt.gca().set_aspect('equal')
        plt.title('Sampled Trajectories')
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Sampled_Trajectories.jpg')
        plt.close()

    def visualize_sampled_goals(self, episode):
        plt.figure()
        dt_sampler = self.dt_sampler
        sampled_goals = dt_sampler.sampled_goals
        
        if self.use_projection and self.background_image is not None:
            plt.imshow(self.background_image, extent=(0, self.background_image.shape[1], self.background_image.shape[0], 0))
            sampled_goals_pixel = np.array([self.calculate_perspective_transform(goal) for goal in sampled_goals])
            plt.xlim(0, self.background_image.shape[1])
            plt.ylim(self.background_image.shape[0], 0)
            plt.axis('off')
        else:
            sampled_goals_pixel = sampled_goals
            plt.grid(True)
            plt.axis('on')
            plt.xlim(self.limits[0][0], self.limits[0][1])
            plt.ylim(self.limits[1][0], self.limits[1][1])
        
        t = np.arange(0, len(sampled_goals_pixel))
        plt.scatter(sampled_goals_pixel[:, 0], sampled_goals_pixel[:, 1], c=t, cmap='viridis', edgecolor='k')
        plt.colorbar(label='Time step')
        plt.gca().set_aspect('equal')
        plt.title('Sampled Goals')
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Sampled_Goals.jpg')
        plt.close()

    def visualize_predicted_states(self, predicted_states, episode):
        plt.figure()
        
        if self.use_projection and self.background_image is not None:
            plt.imshow(self.background_image, extent=(0, self.background_image.shape[1], self.background_image.shape[0], 0))
            predicted_states_pixel = np.array([self.calculate_perspective_transform(state) for state in predicted_states])
            plt.xlim(0, self.background_image.shape[1])
            plt.ylim(self.background_image.shape[0], 0)
            plt.axis('off')
        else:
            predicted_states_pixel = predicted_states
            plt.grid(True)
            plt.axis('on')
            plt.xlim(self.limits[0][0], self.limits[0][1])
            plt.ylim(self.limits[1][0], self.limits[1][1])
        
        plt.scatter(predicted_states_pixel[:, 0], predicted_states_pixel[:, 1], c=np.arange(len(predicted_states_pixel)), cmap='viridis', edgecolor='k')
        plt.title('Proclaimed Trajectory')
        plt.gca().set_aspect('equal')
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Proclaimed_Trajectory.jpg')
        plt.close()


    def visualize_trajectories_on_time(self, episode):
        plt.figure()
        dt_sampler = self.dt_sampler
        x = dt_sampler.latest_achieved[0, :, 0]
        y = dt_sampler.latest_achieved[0, :, 1]
        
        if self.use_projection and self.background_image is not None:
            plt.imshow(self.background_image, extent=(0, self.background_image.shape[1], self.background_image.shape[0], 0))
            coords = np.array([self.calculate_perspective_transform(p) for p in zip(x, y)])
            res_coords = np.array([self.calculate_perspective_transform(rp) for rp in dt_sampler.residual_goals_debug])
            desired_goal = self.calculate_perspective_transform(dt_sampler.latest_desired_goal)
            plt.xlim(0, self.background_image.shape[1])
            plt.ylim(self.background_image.shape[0], 0)
            plt.axis('off')
        else:
            coords = np.array(list(zip(x, y)))
            res_coords = dt_sampler.residual_goals_debug
            desired_goal = dt_sampler.latest_desired_goal
            plt.grid(True)
            plt.axis('on')
            plt.xlim(self.limits[0][0], self.limits[0][1])
            plt.ylim(self.limits[1][0], self.limits[1][1])

        t = np.arange(0, len(coords))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=t, cmap='viridis', edgecolor='k')
        plt.scatter(res_coords[:, 0], res_coords[:, 1], c='white', s=150, marker='*')
        plt.scatter(desired_goal[0], desired_goal[1], color='red', marker='x', s=150, label='Latest Desired Goal')

        plt.colorbar(scatter, label='Time step')
        plt.title('Position Over Time')
        plt.gca().set_aspect('equal')
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Position_Over_Time.jpg')
        plt.close()
    def visualize_trajectories_on_time_on_eval(self, episode, i, states):
        plt.figure()
        dt_sampler = self.dt_sampler

        if self.use_projection and self.background_image is not None:
            plt.imshow(self.background_image, extent=(0, self.background_image.shape[1], self.background_image.shape[0], 0))
            final_goal_projected = self.calculate_perspective_transform(self.dt_sampler.final_goal)
            coords = np.array([self.calculate_perspective_transform(p) for p in states])
            plt.xlim(0, self.background_image.shape[1])
            plt.ylim(self.background_image.shape[0], 0)
            plt.axis('off')
        else:
            final_goal_projected = self.dt_sampler.final_goal
            coords = states
            plt.grid(True)
            plt.axis('on')
            plt.xlim(self.limits[0][0], self.limits[0][1])
            plt.ylim(self.limits[1][0], self.limits[1][1])

        t = np.arange(0, len(coords))
        t_normalized = (t - t.min()) / (t.max() - t.min())

        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=t_normalized, cmap='viridis', edgecolor='k')
        plt.scatter(final_goal_projected[0], final_goal_projected[1], color='red', marker='x', s=150, label='Latest Desired Goal')

        plt.colorbar(scatter, label='Time step')
        plt.title('Position Over Time')
        plt.gca().set_aspect('equal')
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/eval_episode_{episode}_Position_Over_Time_{i}.jpg')
        plt.close()

    def create_combined_np(self, num_points_per_axis=20):
        # Determine if there is a third dimension based on the length of self.limits
        is_3d = len(self.limits) == 3
        
        # Generate evenly spaced points for x and y between their limits
        x_points = np.linspace(self.limits[0][0], self.limits[0][1], num_points_per_axis)
        y_points = np.linspace(self.limits[1][0], self.limits[1][1], num_points_per_axis)
        
        if is_3d:
            # Generate evenly spaced points for z-axis if 3D
            z_points = np.linspace(self.limits[2][0], self.limits[2][1], num_points_per_axis)
            # Create a meshgrid for x, y, and z
            grid_x, grid_y, grid_z = np.meshgrid(x_points, y_points, z_points)
            # Combine the grid into an array of points (x, y, z)
            data_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        else:
            # Create a meshgrid for 2D (x, y)
            grid_x, grid_y = np.meshgrid(x_points, y_points)
            # Combine the grid into an array of points (x, y)
            data_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        return data_points.astype(np.float32)