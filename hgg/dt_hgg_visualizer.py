# visualizer.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class Visualizer:
    def __init__(self, dt_sampler):
        self.dt_sampler = dt_sampler

    def visualize_value_heatmaps_for_debug(self, goals_predicted_during_training):
        dt_sampler = self.dt_sampler
        combined_heatmap = dt_sampler.create_combined_np()
        achieved_values, exploration_values, q_values, _ = dt_sampler.get_rescaled_rewards(combined_heatmap, None)
        probability_loss_preds = dt_sampler.get_probability_loss_over_trajectory(combined_heatmap)
        probability_preds = dt_sampler.get_probabilities_over_trajectory(combined_heatmap)
        
        fig_shape = (4, 3)
        fig, axs = plt.subplots(fig_shape[0], fig_shape[1], figsize=(16, 12), constrained_layout=True)  # 4 rows, 3 columns of subplots
        achieved_values = achieved_values.reshape(-1, 1)
        exploration_values = exploration_values.reshape(-1, 1)
        probability_loss_preds = probability_loss_preds.reshape(-1, 1)
        probability_preds = probability_preds.reshape(-1, 1)
        q_values = q_values.reshape(-1, 1)

        q_pos_val = np.hstack((combined_heatmap, q_values))
        aim_pos_val = np.hstack((combined_heatmap, achieved_values))
        expl_pos_val = np.hstack((combined_heatmap, exploration_values))
        prob_loss_pos_val = np.hstack((combined_heatmap, probability_loss_preds))
        prob_pos_val = np.hstack((combined_heatmap, probability_preds))
        combined_pos_val = np.hstack((combined_heatmap, (q_values + achieved_values + exploration_values)))

        plot_dict = {
            "Q Heatmap": q_pos_val,
            "Aim Heatmap": aim_pos_val,
            "Explore Heatmap": expl_pos_val,
            "Combined Heatmap": combined_pos_val,
            "Probability Loss Heatmap": prob_loss_pos_val,
            "Probability Heatmap": prob_pos_val
        }

        for i, (key, heatmap) in enumerate(plot_dict.items()):
            pos = (i // fig_shape[1], i % fig_shape[1])
            self.plot_heatmap(heatmap, axs[pos[0]][pos[1]], key)
            axs[pos[0]][pos[1]].scatter(goals_predicted_during_training[:, 0], goals_predicted_during_training[:, 1], c=np.arange(len(goals_predicted_during_training)), cmap="gist_heat", s=10)

        i += 1
        pos = (i // fig_shape[1], i % fig_shape[1])
        self.visualize_trajectories_on_time(axs[pos[0]][pos[1]])

        i += 1
        pos = (i // fig_shape[1], i % fig_shape[1])
        self.visualize_trajectories_on_rtgs(axs[pos[0]][pos[1]])

        i += 1
        pos = (i // fig_shape[1], i % fig_shape[1])
        self.visualize_max_rewards(axs[pos[0]][pos[1]])

        i += 1
        pos = (i // fig_shape[1], i % fig_shape[1])
        self.visualize_sampling_points(axs[pos[0]][pos[1]])

        i += 1
        pos = (i // fig_shape[1], i % fig_shape[1])
        self.visualize_sampled_goals(axs[pos[0]][pos[1]])

        plt.savefig(f'{dt_sampler.video_recorder.debug_dir}/combined_heatmaps_episode_{str(dt_sampler.episode)}.jpg')
        plt.close(fig)

    def plot_heatmap(self, data_points, ax, title):
        x = data_points[:, 0]
        y = data_points[:, 1]
        values = data_points[:, 2]
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        grid_values = griddata((x, y), values, (grid_x, grid_y), method='cubic')
        im = ax.imshow(grid_values.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='viridis')
        ax.figure.colorbar(im, ax=ax, label='Value')
        ax.set_title(title)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)

    def visualize_sampled_goals(self, ax):
        dt_sampler = self.dt_sampler
        sampled_goals = dt_sampler.sampled_goals
        t = np.arange(0, len(sampled_goals))
        scatter = ax.scatter(sampled_goals[:, 0], sampled_goals[:, 1], c=t, cmap='viridis', edgecolor='k')
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)

    def visualize_sampling_points(self, ax):
        dt_sampler = self.dt_sampler
        negs = dt_sampler.negatives_buffer.sample(512)
        poss = dt_sampler.positives_buffer.sample(len(dt_sampler.positives_buffer))

        ax.scatter(negs[:, 0], negs[:, 1], c="red")
        ax.scatter(poss[:, 0], poss[:, 1], c="green")

        ax.set_title("sampling points")
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)

    def visualize_max_rewards(self, ax):
        dt_sampler = self.dt_sampler
        max_rewards_np = np.array(dt_sampler.max_rewards_so_far)
        x = np.arange(0, len(max_rewards_np))
        ax.plot(x, max_rewards_np, label='Trend', marker='.', markersize=5, linestyle='-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('reward')
        ax.set_title('Max Rewards')
        ax.grid(True)  # Enable grid for better readability

    def visualize_trajectories_on_time(self, ax, title='Position Over Time'):
        dt_sampler = self.dt_sampler
        x = dt_sampler.latest_achieved[0, :, 0]
        y = dt_sampler.latest_achieved[0, :, 1]
        t = np.arange(0, len(x))

        t_normalized = (t - t.min()) / (t.max() - t.min())
        scatter = ax.scatter(x, y, c=t_normalized, cmap='viridis', edgecolor='k')
        if len(dt_sampler.residual_goals_debug):
            residual_goals_np = np.array(dt_sampler.residual_goals_debug)
            res_x = residual_goals_np[:, 0]
            res_y = residual_goals_np[:, 1]
            ax.scatter(res_x, res_y, color='blue', marker='x', s=100, label='Latest Desired Goal')
        ax.scatter(dt_sampler.latest_desired_goal[0], dt_sampler.latest_desired_goal[1], color='red', marker='x', s=100, label='Latest Desired Goal')

        cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)

    def visualize_trajectories_on_rtgs(self, ax, title='Position Over RTGs'):
        dt_sampler = self.dt_sampler
        x = dt_sampler.latest_achieved[0, :, 0]
        y = dt_sampler.latest_achieved[0, :, 1]
        rtgs = dt_sampler.latest_rtgs[0]

        scatter = ax.scatter(x, y, c=rtgs, cmap='viridis', edgecolor='k')
        ax.scatter(dt_sampler.latest_desired_goal[0], dt_sampler.latest_desired_goal[1], color='red', marker='x', s=100, label='Latest Desired Goal')

        if len(dt_sampler.residual_goals_debug):
            residual_goals_np = np.array(dt_sampler.residual_goals_debug)
            res_x = residual_goals_np[:, 0]
            res_y = residual_goals_np[:, 1]
            ax.scatter(res_x, res_y, color='blue', marker='x', s=100, label='Latest Desired Goal')
        cbar = ax.figure.colorbar(scatter, ax=ax, label='RTG')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)

    def visualize_trajectories_on_qs(self, ax, title='Position Over Qs'):
        dt_sampler = self.dt_sampler
        x = dt_sampler.latest_achieved[0, :, 0]
        y = dt_sampler.latest_achieved[0, :, 1]
        qs = dt_sampler.latest_qs

        scatter = ax.scatter(x, y, c=qs, cmap='viridis', edgecolor='k')
        ax.scatter(dt_sampler.latest_desired_goal[0], dt_sampler.latest_desired_goal[1], color='red', marker='x', s=100, label='Latest Desired Goal')

        if len(dt_sampler.residual_goals_debug):
            residual_goals_np = np.array(dt_sampler.residual_goals_debug)
            res_x = residual_goals_np[:, 0]
            res_y = residual_goals_np[:, 1]
            ax.scatter(res_x, res_y, color='blue', marker='x', s=100, label='Latest Desired Goal')
        cbar = ax.figure.colorbar(scatter, ax=ax, label='Q value')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)
