import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import itertools
from hgg.value_estimator import ValueEstimator
import torch

class Visualizer:
    def __init__(self, dt_sampler):
        self.dt_sampler = dt_sampler
        self.value_estimator: ValueEstimator = dt_sampler.value_estimator
        self.limits = self.dt_sampler.limits
        self.history_of_number_of_states_in_reconstructor = np.array([]).reshape(0, 1)

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

        # Predict time step using the bayesian predictor and plot
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
        dt_sampler = self.dt_sampler
        trajectories = dt_sampler.debug_trajectories
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'cyan', 'magenta', 'brown']

        for i, traj in enumerate(trajectories):
            traj = np.array(traj)
            plt.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)], linewidth=1)
            plt.scatter(traj[:, 0], traj[:, 1], color="grey", edgecolors='k')
            plt.scatter(traj[-1, 0], traj[-1, 1], color='red', edgecolor='k')
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.xlim(self.limits[0][0], self.limits[0][1])
        plt.ylim(self.limits[1][0], self.limits[1][1])
        plt.title('Sampled Trajectories')
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Sampled_Trajectories.jpg')
        plt.close()

    def visualize_sampled_goals(self, episode):
        plt.figure()
        dt_sampler = self.dt_sampler
        sampled_goals = dt_sampler.sampled_goals
        t = np.arange(0, len(sampled_goals))
        plt.scatter(sampled_goals[:, 0], sampled_goals[:, 1], c=t, cmap='viridis', edgecolor='k')
        plt.xlim(self.limits[0][0], self.limits[0][1])
        plt.ylim(self.limits[1][0], self.limits[1][1])
        plt.colorbar(label='Time step')
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.title('Sampled Goals')
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Sampled_Goals.jpg')
        plt.close()

    def visualize_predicted_states(self, predicted_states, episode):
        plt.figure()
        plt.scatter(predicted_states[:, 0], predicted_states[:, 1], c=np.arange(len(predicted_states[:, :])), cmap='viridis', edgecolor='k')
        plt.title('Proclaimed Trajectory')
        plt.xlim(self.limits[0][0], self.limits[0][1])
        plt.ylim(self.limits[1][0], self.limits[1][1])
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.savefig(f'{self.dt_sampler.video_recorder.visualization_dir}/episode_{episode}_Proclaimed_Trajectory.jpg')
        plt.close()

    def visualize_trajectories_on_time(self, episode):
        plt.figure()
        dt_sampler = self.dt_sampler
        x = dt_sampler.latest_achieved[0, :, 0]
        y = dt_sampler.latest_achieved[0, :, 1]
        t = np.arange(0, len(x))
        t_normalized = (t - t.min()) / (t.max() - t.min())
        
        # Save the scatter plot as a mappable object
        scatter = plt.scatter(x, y, c=t_normalized, cmap='viridis', edgecolor='k')
        plt.scatter(dt_sampler.latest_desired_goal[0], dt_sampler.latest_desired_goal[1], color='red', marker='x', s=150, label='Latest Desired Goal')
        
        # Use the scatter object to create the colorbar
        plt.colorbar(scatter, label='Time step')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Position Over Time')
        plt.xlim(self.limits[0][0], self.limits[0][1])
        plt.ylim(self.limits[1][0], self.limits[1][1])
        plt.gca().set_aspect('equal')
        plt.grid(True)
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