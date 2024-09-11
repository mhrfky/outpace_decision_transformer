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
            "PointLongCorridor": {
                "background_image": os.path.join(base_image_path, "PointLongCorridor_background.png"),
                "world_coordinate": np.float32([[0, 0], [0, 12], [24, 12], [24, 0]]),
                "image_coordinate": np.float32([[85, 305], [85, 150], [395, 150], [395, 305]])
            }
        }

        # Retrieve the background image and transformation matrix using env_name
        env_name = dt_sampler.env_name
        if env_name in self.env_settings:
            settings = self.env_settings[env_name]
            self.background_image = mpimg.imread(settings["background_image"])
            self.M = cv2.getPerspectiveTransform(settings["world_coordinate"], settings["image_coordinate"])
        else:
            raise ValueError(f"Environment '{env_name}' not found in the predefined settings.")

    def calculate_perspective_transform(self, world_position):
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

        combined_heatmap = self.create_combined_np()
        total_values, achieved_values, exploration_values, q_values = value_estimator.get_state_values(combined_heatmap, None)

        achieved_values = achieved_values.reshape(-1, 1)
        q_values = q_values.reshape(-1, 1)
        total_values = total_values.reshape(-1, 1)
        q_pos_val = np.hstack((combined_heatmap, q_values))
        aim_pos_val = np.hstack((combined_heatmap, achieved_values))

        # Plotting each figure separately and saving
        self.plot_and_save(q_pos_val, 'Q Heatmap', dt_sampler.episode)
        self.plot_and_save(aim_pos_val, 'Aim Heatmap', dt_sampler.episode)

        self.visualize_trajectories_on_time(dt_sampler.episode)
        self.visualize_sampled_goals(dt_sampler.episode)
        self.visualize_sampled_trajectories(dt_sampler.episode)
        self.visualize_predicted_states(predicted_states, dt_sampler.episode)

        # Predict time step using the Bayesian predictor and plot
        predictor = self.dt_sampler.bayesian_predictor
        combined_heatmap_t = torch.tensor(combined_heatmap, device="cuda", dtype=torch.float32)
        t = predictor.predict(combined_heatmap_t)
        timestep_predictions = np.hstack((combined_heatmap, t.reshape(-1, 1)))
        self.plot_and_save(timestep_predictions, 'Time Step Prediction', dt_sampler.episode)

    def plot_and_save(self, data_points, title, episode):
        plt.figure()
        x = data_points[:, 0]
        y = data_points[:, 1]
        values = data_points[:, 2]
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        grid_values = griddata((x, y), values, (grid_x, grid_y), method='cubic')
        plt.imshow(grid_values.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(title)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.xlim(self.limits[0][0], self.limits[0][1])
        plt.ylim(self.limits[1][0], self.limits[1][1])
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_{title}.jpg')
        plt.close()

    def visualize_sampled_trajectories(self, episode):
        plt.figure()
        plt.imshow(self.background_image, extent=(0, self.background_image.shape[1], self.background_image.shape[0], 0))
        dt_sampler = self.dt_sampler
        trajectories = dt_sampler.debug_trajectories
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'cyan', 'magenta', 'brown']

        for i, traj in enumerate(trajectories):
            traj = np.array([self.calculate_perspective_transform(p) for p in traj])  # Apply transformation
            plt.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)], linewidth=1)
            plt.scatter(traj[:, 0], traj[:, 1], color="grey", edgecolors='k')  # Original size scatter points
            plt.scatter(traj[-1, 0], traj[-1, 1], color='red', edgecolor='k', s=50)  # Keep 'x' mark size
        plt.gca().set_aspect('equal')
        plt.xlim(0, self.background_image.shape[1])
        plt.ylim(self.background_image.shape[0], 0)
        plt.title('Sampled Trajectories')
        plt.axis('off')  # Turn off the axis
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Sampled_Trajectories.jpg')
        plt.close()

    def visualize_sampled_goals(self, episode):
        plt.figure()
        plt.imshow(self.background_image, extent=(0, self.background_image.shape[1], self.background_image.shape[0], 0))
        dt_sampler = self.dt_sampler
        sampled_goals = dt_sampler.sampled_goals
        sampled_goals_pixel = np.array([self.calculate_perspective_transform(goal) for goal in sampled_goals])  # Transform goals
        t = np.arange(0, len(sampled_goals_pixel))
        plt.scatter(sampled_goals_pixel[:, 0], sampled_goals_pixel[:, 1], c=t, cmap='viridis', edgecolor='k')  # Original size scatter points
        plt.xlim(0, self.background_image.shape[1])
        plt.ylim(self.background_image.shape[0], 0)
        plt.colorbar(label='Time step')
        plt.gca().set_aspect('equal')
        plt.title('Sampled Goals')
        plt.axis('off')  # Turn off the axis
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Sampled_Goals.jpg')
        plt.close()

    def visualize_predicted_states(self, predicted_states, episode):
        plt.figure()
        plt.imshow(self.background_image, extent=(0, self.background_image.shape[1], self.background_image.shape[0], 0))
        predicted_states_pixel = np.array([self.calculate_perspective_transform(state) for state in predicted_states])  # Transform states
        plt.scatter(predicted_states_pixel[:, 0], predicted_states_pixel[:, 1], c=np.arange(len(predicted_states_pixel)), cmap='viridis', edgecolor='k')  # Original size scatter points
        plt.title('Proclaimed Trajectory')
        plt.xlim(0, self.background_image.shape[1])
        plt.ylim(self.background_image.shape[0], 0)
        plt.gca().set_aspect('equal')
        plt.axis('off')  # Turn off the axis
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Proclaimed_Trajectory.jpg')
        plt.close()

    def visualize_trajectories_on_time(self, episode):
        plt.figure()
        plt.imshow(self.background_image, extent=(0, self.background_image.shape[1], self.background_image.shape[0], 0))
        dt_sampler = self.dt_sampler
        x = dt_sampler.latest_achieved[0, :, 0]
        y = dt_sampler.latest_achieved[0, :, 1]
        coords = np.array([self.calculate_perspective_transform(p) for p in zip(x, y)])  # Transform coordinates
        t = np.arange(0, len(coords))
        t_normalized = (t - t.min()) / (t.max() - t.min())

        res_coords = np.array([self.calculate_perspective_transform(rp) for rp in dt_sampler.residual_goals_debug])  # Transform coordinates

        

        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=t_normalized, cmap='viridis', edgecolor='k')  # Original size scatter points
        plt.scatter(res_coords[:, 0], res_coords[:, 1], c='white', s=150,marker='*' )  # Original size scatter points
        plt.scatter(*self.calculate_perspective_transform(dt_sampler.latest_desired_goal), color='red', marker='x', s=150, label='Latest Desired Goal')  # Original size 'x' mark

        plt.colorbar(scatter, label='Time step')
        plt.title('Position Over Time')
        plt.xlim(0, self.background_image.shape[1])
        plt.ylim(self.background_image.shape[0], 0)
        plt.gca().set_aspect('equal')
        plt.axis('off')  # Turn off the axis
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Position_Over_Time.jpg')
        plt.close()

    def create_combined_np(self):
        data_points = [
            [x, y]
            for x, y in itertools.product(
                range(self.limits[0][0], self.limits[0][1] + 1),
                range(self.limits[1][0], self.limits[1][1] + 1),
            )
        ]
        return np.array(data_points, dtype=np.float32)