# visualizer.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import itertools
from hgg.value_estimator import ValueEstimator
from playground2 import time_decorator
import torch
class Visualizer:
    def __init__(self, dt_sampler):
        self.dt_sampler  = dt_sampler
        self.value_estimator : ValueEstimator  = dt_sampler.value_estimator
        self.limits = self.dt_sampler.limits
        self.history_of_number_of_states_in_reconstructor = np.array([]).reshape(0, 1)
    
    def visualize_value_heatmaps_for_debug(self, goals_predicted_during_training, predicted_states, predicted_rtgs):
        dt_sampler = self.dt_sampler
        value_estimator = self.value_estimator

        combined_heatmap = self.create_combined_np()
        total_values, achieved_values, exploration_values, q_values = value_estimator.get_state_values(combined_heatmap, None)
        
        fig_shape = (4, 3)
        fig, axs = plt.subplots(fig_shape[0], fig_shape[1], figsize=(16, 12), constrained_layout=True)  # 4 rows, 3 columns of subplots
        achieved_values = achieved_values.reshape(-1, 1)
        exploration_values = exploration_values.reshape(-1, 1)
        q_values = q_values.reshape(-1, 1)
        total_values = total_values.reshape(-1, 1)
        q_pos_val = np.hstack((combined_heatmap, q_values))
        aim_pos_val = np.hstack((combined_heatmap, achieved_values))
        expl_pos_val = np.hstack((combined_heatmap, exploration_values))
        combined_pos_val = np.hstack((combined_heatmap, total_values))

        plot_dict = {
            "Q Heatmap": q_pos_val,
            "Aim Heatmap": aim_pos_val,
            "Explore Heatmap": expl_pos_val,
            "Combined Heatmap": combined_pos_val,
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
        self.visualize_sampled_goals(axs[pos[0]][pos[1]])

        i += 1
        pos = (i // fig_shape[1], i % fig_shape[1])
        self.visualize_sampled_trajectories(axs[pos[0]][pos[1]])

        # i += 1
        # pos = (i // fig_shape[1], i % fig_shape[1])
        # ax = axs[pos[0]][pos[1]]
        # self.history_of_number_of_states_in_reconstructor = np.vstack((self.history_of_number_of_states_in_reconstructor, np.array([len(dt_sampler.trajectory_reconstructor.states)])))
        # x = np.arange(0, len( self.history_of_number_of_states_in_reconstructor ))
        # ax.plot(x,  self.history_of_number_of_states_in_reconstructor , label='Trend', marker='.', markersize=5, linestyle='-', linewidth=2)
        # ax.set_xlabel('x')
        # ax.set_ylabel('reward')
        # ax.set_title('Max Rewards')

        i += 1
        pos = (i // fig_shape[1], i % fig_shape[1])
        ax = axs[pos[0]][pos[1]]
        ax.scatter(predicted_states[:20, 0], predicted_states[:20, 1], c=np.arange(len(predicted_states[:20,:])), cmap='viridis', edgecolor='k')
        ax.scatter(predicted_states[20:, 0], predicted_states[20:, 1], c=np.arange(len(predicted_states[20:,:])), cmap='viridis', edgecolor='k')

        ax.set_title('Proclaimed Trajectory')
        ax.set_xlim(self.limits[0][0], self.limits[0][1])
        ax.set_ylim(self.limits[1][0], self.limits[1][1])
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)
        # traj_lens = self.dt_sampler.debug_traj_lens
        # ax.hist(traj_lens, bins=40, range=(0, 40))
        # ax.set_xlabel('Trajectory Length')
        # ax.set_ylabel('Frequency')
        # ax.set_title('Histogram of Trajectory Lengths')
        # q_uncertainties = []
        # q1s = []
        # q2s = []
        # for state in dt_sampler.trajectory_reconstructor.states:
        #     q1, q2  = value_estimator.get_q_values(torch.tensor(state, device="cuda", dtype=torch.float32))
        #     q1 = q1.cpu().detach().numpy()
        #     q2 = q2.cpu().detach().numpy()
        #     q_uncertainty = np.abs(q1 - q2)
        #     q_uncertainties.append(q_uncertainty)
        #     q1s.append(q1)
        #     q2s.append(q2)
        # q_uncertainties = np.array(q_uncertainties).squeeze(-1).squeeze(-1)
        # q1s = np.array(q1s).squeeze(-1).squeeze(-1)
        # q2s = np.array(q2s).squeeze(-1).squeeze(-1)
        
        i += 1
        pos = (i // fig_shape[1], i % fig_shape[1])
        ax = axs[pos[0]][pos[1]]
        # entropy_list =[]
        # for elem in combined_heatmap:
        #     entropy_list.append(dt_sampler.entropy_gain(elem))
        # entropy_list = np.array(entropy_list)
        # scatter = ax.scatter(combined_heatmap[:, 0], combined_heatmap[:, 1], c=entropy_list, cmap='viridis', edgecolor='k')
        # ax.set_title('Entropy Gain')
        # cbar = ax.figure.colorbar(scatter, ax=ax, label='Entropy Gain')
        # ax.set_xlim(-2, 10)
        # ax.set_ylim(-2, 10)
        # ax.set_aspect('equal')  # Ensuring equal aspect ratio
        # ax.grid(True)

        

        i += 1
        pos = (i // fig_shape[1], i % fig_shape[1])
        ax = axs[pos[0]][pos[1]]
        # scatter = ax.scatter(dt_sampler.trajectory_reconstructor.states[:,0], dt_sampler.trajectory_reconstructor.states[:,1], c=q2s, cmap='viridis', edgecolor='k')
        # cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')
        # ax.set_title('Q 2')
        # ax.set_xlim(-2, 10)
        # ax.set_ylim(-2, 10)
        # ax.set_aspect('equal')  # Ensuring equal aspect ratio
        # ax.grid(True)
        

        # self.plot_residuals_if_exists(axs[pos[0]][pos[1]])

        i += 1
        pos = (i // fig_shape[1], i % fig_shape[1])
        ax = axs[pos[0]][pos[1]]
        # scatter = ax.scatter(dt_sampler.trajectory_reconstructor.states[:,0], dt_sampler.trajectory_reconstructor.states[:,1], c=q_uncertainties, cmap='viridis', edgecolor='k')
        # cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')
        # ax.set_title('Q Uncertainty')
        # ax.set_xlim(-2, 10)
        # ax.set_ylim(-2, 10)
        # ax.set_aspect('equal')  # Ensuring equal aspect ratio
        # ax.grid(True)

        

        plt.savefig(f'{dt_sampler.video_recorder.visualization_dir}/combined_heatmaps_episode_{str(dt_sampler.episode)}.jpg')
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
        ax.set_xlim(self.limits[0][0], self.limits[0][1])
        ax.set_ylim(self.limits[1][0], self.limits[1][1])
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)
    def visualize_sampled_trajectories(self,ax):
        dt_sampler = self.dt_sampler
        x = dt_sampler.trajectory_reconstructor.states[ :, 0]
        y = dt_sampler.trajectory_reconstructor.states[ :, 1]
        scatter = ax.scatter(x, y, c = 'grey', edgecolor='k')
        trajectories = dt_sampler.debug_trajectories
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'cyan', 'magenta', 'brown']

        for i, traj in enumerate(trajectories):
            x = traj[:,0]
            y = traj[:,1]
            ax.plot(x , y, color=colors[i % len(colors)], linewidth=1)
            ax.scatter(traj[-1,0], traj[-1,1], color = 'red', edgecolor = 'k')
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)
        ax.set_xlim(self.limits[0][0], self.limits[0][1])
        ax.set_ylim(self.limits[1][0], self.limits[1][1])
        ax.set_title('Sampled Trajectories')
    def visualize_sampled_goals(self, ax):
        dt_sampler = self.dt_sampler
        sampled_goals = dt_sampler.sampled_goals
        t = np.arange(0, len(sampled_goals))
        scatter = ax.scatter(sampled_goals[:, 0], sampled_goals[:, 1], c=t, cmap='viridis', edgecolor='k')
        ax.set_xlim(self.limits[0][0], self.limits[0][1])
        ax.set_ylim(self.limits[1][0], self.limits[1][1])
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
        ax.set_xlim(self.limits[0][0], self.limits[0][1])
        ax.set_ylim(self.limits[1][0], self.limits[1][1])
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
      

            # ax.scatter(sampled_x, sampled_y, c=sampled_t, cmap = "plasma", marker='+', s=100)
        ax.scatter(dt_sampler.latest_desired_goal[0], dt_sampler.latest_desired_goal[1], color='red', marker='x', s=100, label='Latest Desired Goal')
        for res_goal in self.dt_sampler.residual_goals_debug:
            ax.scatter(res_goal[0], res_goal[1], color='orange', marker='*', s=100, label='Residual Goal')
        circle = plt.Circle((dt_sampler.latest_desired_goal[0], dt_sampler.latest_desired_goal[1]), 0.5, fill=False, edgecolor='black')
        ax.add_patch(circle)
        cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.set_xlim(self.limits[0][0], self.limits[0][1])
        ax.set_ylim(self.limits[1][0], self.limits[1][1])
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)

    def visualize_trajectories_on_rtgs(self, ax, title='Position Over RTGs'):
        dt_sampler = self.dt_sampler
        x = dt_sampler.latest_achieved[0, :, 0]
        y = dt_sampler.latest_achieved[0, :, 1]
        rtgs = dt_sampler.latest_rtgs[0]

        scatter = ax.scatter(x, y, c=rtgs, cmap='viridis', edgecolor='k')
        ax.scatter(dt_sampler.latest_desired_goal[0], dt_sampler.latest_desired_goal[1], color='red', marker='x', s=100, label='Latest Desired Goal')

        cbar = ax.figure.colorbar(scatter, ax=ax, label='RTG')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.set_xlim(self.limits[0][0], self.limits[0][1])
        ax.set_ylim(self.limits[1][0], self.limits[1][1])
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)
    def plot_residuals_if_exists(self, ax):
        if len(self.dt_sampler.residual_goals_debug):
            achieved_states_np = np.array(self.dt_sampler.residual_goals_debug)[0]
            residual_goals_np = np.array(self.dt_sampler.residual_goals_debug)[1]
            ax.scatter(self.dt_sampler.latest_desired_goal[0], self.dt_sampler.latest_desired_goal[1], c='green', s = 100, marker = '+', label='Achieved States')
            ax.scatter(achieved_states_np[:, 0], achieved_states_np[:, 1], c='blue', s = 20, label='Achieved States')
            ax.scatter(residual_goals_np[:, 0], residual_goals_np[:, 1], c = np.arange(len(residual_goals_np)), s = 5, cmap='viridis', label='Residual Goals')
            ax.scatter(residual_goals_np[-1, 0], residual_goals_np[-1, 1], color='red', marker = 'x', s=100, label='Final Residual Goal')
            ax.set_title('Residual Trajectory')
            ax.set_xlim(self.limits[0][0], self.limits[0][1])
            ax.set_ylim(self.limits[1][0], self.limits[1][1])
            ax.set_aspect('equal')  # Ensuring equal aspect ratio
            ax.grid(True)
    def plot_residuals_till_now(self,ax):
        x = self.dt_sampler.residuals_till_now[:,0]
        y = self.dt_sampler.residuals_till_now[:,1]
        ax.scatter(x, y, c = np.arange(len(x)), cmap='viridis', label='Residual Goals', edgecolor='k')
        ax.set_title('Residual Goals Till Now')
        ax.set_xlim(self.limits[0][0], self.limits[0][1])
        ax.set_ylim(self.limits[1][0], self.limits[1][1])
        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)

    def create_combined_np(self):
        data_points = [
            [x, y]
            for x, y in itertools.product(
                range(self.limits[0][0], self.limits[0][1]),
                range(self.limits[1][0], self.limits[1][1]),
            )
        ]
        return np.array(data_points, dtype=np.float32)