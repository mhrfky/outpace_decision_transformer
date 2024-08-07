import numpy as np
import torch
import torch.nn.functional as F
from dt.models.decision_transformer import DecisionTransformer
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from video import VideoRecorder

from hgg.dt_hgg_visualizer import Visualizer
from hgg.utils import trajectory_similarity_loss
from hgg.value_estimator import ValueEstimator
from hgg.states_buffer import StatesBuffer
from hgg.trajectory_buffer import TrajectoryBuffer

from video import VideoRecorder
from hgg.trajectory_reconstructor import TrajectoryReconstructor
from hgg.utils import calculate_max_distance
from playground2 import time_decorator
from scipy.interpolate import splprep, splev
from debug_utils import plot_positions, plot_two_array_positions
debug_loc_list = []
debug_rtg_list = []
class DTSampler:    
	def __init__(self, goal_env, goal_eval_env, 				
				agent = None,
				add_noise_to_goal= False, beta = 1, gamma=-1, sigma = 1, device = 'cuda', critic = None, 
				dt : DecisionTransformer = None, rtg_optimizer = None, state_optimizer = None,
				loss_fn  = torch.nn.MSELoss(),
				video_recorder : VideoRecorder = None
				):
		# Assume goal env
		self.env = goal_env
		self.eval_env = goal_eval_env
		
		self.add_noise_to_goal = add_noise_to_goal
		self.agent = agent
		self.state_optimizer = state_optimizer
		self.critic = critic

		self.gamma = -1
		self.beta = 1 # q values
		self.sigma = 1 # exploration

		self.device = device
		self.num_seed_steps = 2000	 # TODO init this from config later
		self.success_threshold = {'AntMazeSmall-v0' : 1.0, # 0.5,
								  'PointUMaze-v0' : 0.5,
		  						  'PointNMaze-v0' : 0.5,
								  'sawyer_peg_push' : getattr(self.env, 'TARGET_RADIUS', None),
								  'sawyer_peg_pick_and_place' : getattr(self.env, 'TARGET_RADIUS', None),
								  'PointSpiralMaze-v0' : 0.5,								  
								}

		self.dim = np.prod(self.env.convert_obs_to_dict(self.env.reset())['achieved_goal'].shape)
		self.video_recorder : VideoRecorder = video_recorder
		
		self.init_goal = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['achieved_goal'].copy()
		
		self.path_similarity_loss = trajectory_similarity_loss
		self.rtg_pred_loss = torch.nn.L1Loss()
		self.goal_val_loss = torch.nn.L1Loss()

		self.final_goal = np.array([0,8]) #TODO make this parametrized
		self.latest_achieved = None
		self.dt = dt
		self.max_achieved_reward = 0
		self.limits = [[-2,10],[-2,10], [0,0]] # TODO make this generic
		self.latest_desired_goal = self.final_goal
		self.max_rewards_so_far = []
		self.residual_goals_debug = np.array([]).reshape(0,2)
		self.number_of_trajectories_per_episode = 2
		
		self.sampled_goals = np.array([]).reshape(0,2)
		self.sampled_states = np.array([]).reshape(0,2)
		self.trajectory_buffer = TrajectoryBuffer(100)
		self.trajectory_reconstructor = TrajectoryReconstructor()

		self.value_estimator = ValueEstimator(self.agent, self.eval_env, self.final_goal, self.num_seed_steps, self.init_goal, gamma, beta, sigma , rescale= True)
		self.visualizer = Visualizer(self)	
		self.number_of_iterations = 60
		self.debug_trajectories = []
		self.debug_traj_lens = []
		self.residual_goal_rtg_increase = 1
		self.log_every_n_times =  1
		self.residual_this_episode = False
		self.residuals_till_now = np.array([]).reshape(0,2)
	def reward_to_rtg(self,rewards):
		return rewards - rewards[-1]

	@time_decorator
	def update(self, step, episode, achieved_goals, actions, qs):
			
		self.step = step
		self.episode = episode
		self.debug_trajectories = []
		achieved_states = np.array([self.eval_env.convert_obs_to_dict(achieved_goals[i])["achieved_goal"] for i in range(len(achieved_goals))])
		actions = np.array(actions)

		if not len(qs):
			qs = np.zeros((100,2), dtype=float)
		rewards, achieved_values, exploration_values, q_values = self.value_estimator.get_state_values(achieved_states, qs)
		rtgs = self.reward_to_rtg(rewards)

		debug_loc_list.append(achieved_states)
		debug_rtg_list.append(rtgs)

		self.latest_achieved 	= np.array([achieved_states])
		self.latest_acts        = np.array([actions])
		self.latest_rtgs 		= np.array([rtgs ])

		
		self.max_achieved_reward = max(max(rewards),self.max_achieved_reward)
		self.latest_qs = q_values

		goals_predicted_debug_np = self.train(achieved_states, rtgs) #visualize this
		if self.episode % self.log_every_n_times == 0:
			proclaimed_states, _ , proclaimed_rtgs, _ = self.generate_next_n_states(self.latest_achieved[:,:20], self.latest_acts[:,:20], np.expand_dims(self.latest_rtgs[:,:20],axis=-1), torch.arange(20, device="cuda").unsqueeze(0), n = 90)
			proclaimed_states = proclaimed_states.squeeze(0).detach().cpu().numpy() 
			proclaimed_rtgs = proclaimed_rtgs.squeeze(0).detach().cpu().numpy()
			self.visualize_value_heatmaps_for_debug(goals_predicted_debug_np, proclaimed_states, proclaimed_rtgs)
			self.residual_this_episode = False
		self.max_rewards_so_far.append(max(rewards))

		self.residual_goals_debug = np.array([]).reshape(0,2)
		self.sampled_states = np.array([]).reshape(0,2)
		self.debug_traj_lens = []

		# self.sampled_goals = np.array([]).reshape(0,2)
	def cubic_spline_interpolation_loss(self, trajectory_t , residuals_t):
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
		
	def entropy_gain(self, new_goal, bandwidth=0.1):
		current_goals = self.trajectory_reconstructor.states
		current_goals_tensor = torch.tensor(current_goals, dtype=torch.float32)
		new_goal_tensor = torch.tensor(new_goal, dtype=torch.float32).unsqueeze(0)
		current_entropy = self.calculate_entropy(current_goals_tensor, bandwidth)
		updated_goals = torch.cat([current_goals_tensor, new_goal_tensor], dim=0)
		updated_entropy = self.calculate_entropy(updated_goals, bandwidth)
		return updated_entropy - current_entropy		
	def calculate_entropy(self,goals, bandwidth=0.1):
		goals_tensor = torch.tensor(goals, dtype=torch.float32)
		kde = torch.distributions.MultivariateNormal(loc=goals_tensor, covariance_matrix=bandwidth * torch.eye(goals_tensor.size(1)))
		log_density = kde.log_prob(goals_tensor)
		density = torch.exp(log_density)
		entropy = -torch.sum(density * log_density)
		return entropy

	@time_decorator
	def train(self, achieved_states, rtgs):
		goals_predicted_debug_np = np.array([[0,0]])
		max_distance = calculate_max_distance(achieved_states)	
		self.residual_goal_length = 5
		for trajectory, rtgs_t, traj_len in self.trajectory_reconstructor.get_shortest_path_trajectories_with_yield(achieved_states, rtgs, top_n=50, eval_fn = self.value_estimator.get_state_values, n= 10, max_distance=max_distance):
			self.debug_trajectories.append(trajectory[-traj_len:])
			self.debug_traj_lens.append(traj_len)
			trajectory_t = torch.tensor(trajectory, device="cuda", dtype=torch.float32).unsqueeze(0)
			rtgs_t = torch.tensor(rtgs_t, device="cuda", dtype=torch.float32).unsqueeze(0)
			actions_t = torch.zeros((1, trajectory_t.size(1), 2), device="cuda", dtype=torch.float32)
			timesteps_t = torch.arange(trajectory_t.size(1), device="cuda").unsqueeze(0)

			input_achieved_t = trajectory_t[:, :-traj_len, :]
			input_actions_t = actions_t[:, :-traj_len, :]
			input_rtgs_t = rtgs_t[:, :-traj_len]
			input_timesteps_t = timesteps_t[:, :-traj_len]

			input_rtgs_t = input_rtgs_t - rtgs_t[0, -1]
			input_rtgs_t = input_rtgs_t.unsqueeze(-1)

			generated_states_t, generated_actions, generated_rtgs_t, generated_timesteps = self.generate_next_n_states(
				input_achieved_t,input_actions_t,input_rtgs_t,input_timesteps_t, n=traj_len + self.residual_goal_length) 
			generated_states_t, residual_states_t = np.split(generated_states_t,[-self.residual_goal_length], axis= 1)
			generated_rtgs_t = generated_rtgs_t[:,:-self.residual_goal_length]
			entropy_gains_t = torch.tensor([self.entropy_gain(goal) for goal in residual_states_t.squeeze(0).detach().cpu().numpy()]).sum()
			goals_predicted_debug_np = np.vstack((goals_predicted_debug_np, generated_states_t.squeeze(0)[-traj_len:].detach().cpu().numpy()))

			state_values_t, achieved_values_t, exploration_values_t, q_values_t = self.value_estimator.get_state_values_t(generated_states_t[:,-traj_len-1:,:])
			rtg_values_t = self.reward_to_rtg(state_values_t)

			goal_val_loss = torch.nn.L1Loss()(rtg_values_t[:,1:], rtgs_t[:,-traj_len:])
			rtg_pred_loss = torch.nn.L1Loss()(generated_rtgs_t[:,-traj_len:].squeeze(-1), rtg_values_t[:,1:])
			similarity_loss, dtw_distance, euclidean_distance, smoothness_reg =  trajectory_similarity_loss(generated_states_t[:,-traj_len:,:].squeeze(0), trajectory_t[0,-traj_len:,:])

			# residual_goals_similarity_loss = self.cubic_spline_interpolation_loss(generated_states_t[:,-traj_len:,:], residual_states_t)
			q_gain_rewards_t = torch.diff(q_values_t)
			state_val_gain_rewards_t =  torch.diff(state_values_t)
			# 1 + 0.4 + 1 + 1 + 2
			total_loss = -entropy_gains_t  + dtw_distance + smoothness_reg * 0.3 + rtg_pred_loss + goal_val_loss -  2* torch.mean(state_val_gain_rewards_t)#dtw_distance +  0.4 * smoothness_reg  +  rtg_pred_loss +   goal_val_loss - q_values_t.mean()
			self.state_optimizer.zero_grad()
			total_loss.backward(retain_graph=True)
			torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
			self.state_optimizer.step()
		# print(f"episode : {self.episode}, {len(self.debug_traj_lens)} trajectories generated and trained on this iteration")
		return goals_predicted_debug_np
	@time_decorator
	def generate_next_n_states(self, achieved_states_t, actions_t, rtgs_t, timesteps_t, n=10):
		if not isinstance(achieved_states_t, torch.Tensor):
			achieved_states_t = torch.tensor(achieved_states_t, device="cuda", dtype=torch.float32)
		if not isinstance(actions_t, torch.Tensor):
			actions_t = torch.tensor(actions_t, device="cuda", dtype=torch.float32)
		if not isinstance(rtgs_t, torch.Tensor):
			rtgs_t = torch.tensor(rtgs_t, device="cuda", dtype=torch.float32)
		if not isinstance(timesteps_t, torch.Tensor):
			timesteps_t = torch.tensor(timesteps_t, device="cuda", dtype=torch.float32)
		for _ in range(n):
			
			predicted_state_mean_t, predicted_state_logvar, predicted_action_t, predicted_rtg_t = self.dt.forward(
				achieved_states_t, actions_t, None, rtgs_t, timesteps_t)

			predicted_state_mean_t = predicted_state_mean_t[:, -1, :].unsqueeze(0)  # Take the last predicted step
			predicted_action_t = predicted_action_t[:, -1, :].unsqueeze(0)
			predicted_rtg_t = predicted_rtg_t[:, -1, :].unsqueeze(0)

			achieved_states_t = torch.concat((achieved_states_t, predicted_state_mean_t), dim = 1)
			actions_t = torch.concat((actions_t, predicted_action_t), dim = 1)
			rtgs_t = torch.concat((rtgs_t, predicted_rtg_t), dim = 1)
			timesteps_t = torch.concat((timesteps_t, torch.tensor([timesteps_t[0][-1] + 1], device="cuda").unsqueeze(0)), dim = 1)

		return achieved_states_t, actions_t, rtgs_t, timesteps_t
	@time_decorator
	def sample(self, episode_observes = None, episode_acts = None, qs = None):
		if episode_observes is None or qs is None or episode_acts is None:
			if self.latest_achieved is None:
				return np.array([0,8])
			# goal_t =  self.generate_goal(self.latest_achieved,self.latest_rtgs + self.return_to_add)
			actions_t = torch.tensor(self.latest_acts, device="cuda", dtype=torch.float32)
			rtgs_t = torch.tensor(self.latest_rtgs, device="cuda", dtype=torch.float32).unsqueeze(2)
			achieved_goals_states_t = torch.tensor(self.latest_achieved, device="cuda", dtype=torch.float32)
			generated_states_t, _,_,_ = 	self.generate_next_n_states(achieved_goals_states_t, actions_t, rtgs_t, torch.arange(achieved_goals_states_t.shape[1], device="cuda").unsqueeze(0), n = 10)

			generated_states = generated_states_t.squeeze(0).detach().cpu().numpy()
			goal = generated_states[-1]

			self.latest_desired_goal = goal
			self.sampled_goals = np.concatenate((self.sampled_goals, [goal]))
			self.sampled_states = np.concatenate((self.sampled_states, generated_states[-10:]))
			return goal
		else:
			states = np.array([self.eval_env.convert_obs_to_dict(episode_observes[i])["achieved_goal"] for i in range(len(episode_observes))])
			actions = np.array(episode_acts)

			rewards, achieved_values, exploration_values, q_values = self.value_estimator.get_state_values(states, qs)
			rtgs = self.reward_to_rtg(rewards)
   
			achieved_goals_states_t = torch.tensor([states], device="cuda", dtype=torch.float32)
			rtgs_t = torch.tensor([rtgs], device="cuda", dtype=torch.float32).unsqueeze(2) + self.residual_goal_rtg_increase
			actions_t = torch.tensor([actions], device="cuda", dtype=torch.float32)#.unsqueeze(0)
			n = 110 - achieved_goals_states_t.shape[1]
			generated_states_t, _,_,_ = self.generate_next_n_states(achieved_goals_states_t, actions_t, rtgs_t, torch.arange(achieved_goals_states_t.shape[1], device="cuda").unsqueeze(0), n = n)
			generated_states = generated_states_t.squeeze(0).detach().cpu().numpy()
			goal = generated_states[-1]
			self.residual_this_episode = True
			self.residual_goals_debug = np.vstack((self.residual_goals_debug, goal))
			self.residuals_till_now = np.vstack((self.residuals_till_now, [goal]))
			print(f"Residual goal generated in episode: {self.episode} in the step : {110 -n }")

			return goal 



	@time_decorator
	def visualize_value_heatmaps_for_debug(self, goals_predicted_during_training, proclaimed_states, proclaimed_rtgs):
		self.visualizer.visualize_value_heatmaps_for_debug(goals_predicted_during_training, proclaimed_states, proclaimed_rtgs)

