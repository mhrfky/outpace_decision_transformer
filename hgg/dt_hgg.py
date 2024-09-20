import numpy as np
import torch
import torch.nn.functional as F
from dt.models.decision_transformer import DecisionTransformer
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from video import VideoRecorder

from hgg.dt_hgg_visualizer import Visualizer
from hgg.value_estimator import ValueEstimator
from hgg.states_buffer import StatesBuffer
from hgg.trajectory_buffer import TrajectoryBuffer

from video import VideoRecorder
from hgg.trajectory_reconstructor import TrajectoryReconstructor
from hgg.utils import calculate_max_distance
from debug_utils import time_decorator
from debug_utils import plot_positions, plot_two_array_positions
from hgg.loss_utils import  trajectory_similarity_loss
from sklearn.cluster import KMeans
from hgg.utils import reorder_centroids_dtw
from hgg.trajectory_bufferer import TrajectoryBufferer
from hgg.bayesian_predictor import BayesianPredictor
from hgg.kmeans_regulated_subtraj_buffer import KMeansRegulatedSubtrajBuffer as Km
from hgg.loss_utils import euclidean_distance_loss
debug_loc_list = []
debug_rtg_list = []
class DTSampler:    
	def __init__(self, goal_env, goal_eval_env, 				
				agent = None,
				add_noise_to_goal= False, beta = 2, gamma=-1, sigma = 1, device = 'cuda',
				dt : DecisionTransformer = None,  optimizer = None,
				video_recorder : VideoRecorder = None, rescale_rewards = True, env_name = "PointUMaze-v0", max_ep_length=128, final_goal = np.array([0., 8.]), limits = np.array([[-10, 10], [-10, 10]])
				):
		self.limits = limits
		self.final_goal = final_goal
		self.state_dim = self.final_goal.shape[0]
		self.n = 4
		self.max_ep_length = max_ep_length
		self.eval_env = goal_eval_env
		self.dt = dt
		self.device = device
		self.num_seed_steps = 2000	 # TODO init this from config later
		self.init_goal = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['achieved_goal'].copy()
		self.agent = agent
		self.state_optimizer = optimizer
		self.value_estimator = ValueEstimator(self.agent, self.eval_env, self.final_goal, self.num_seed_steps, self.init_goal, gamma, beta, sigma , rescale= rescale_rewards)
		self.dim = np.prod(self.eval_env.convert_obs_to_dict(self.eval_env.reset())['achieved_goal'].shape)#TODO use this instead
		self.env_name = env_name
		self.add_noise_to_goal = add_noise_to_goal
		
		# Trajectory Buffers
		self.trajectory_buffer = TrajectoryBuffer(100)
		self.trajectory_reconstructor = TrajectoryReconstructor(merge_when_full=False, final_goal=self.final_goal)

		# Losses
		self.path_similarity_loss = trajectory_similarity_loss
		self.rtg_pred_loss = torch.nn.L1Loss()
		self.goal_val_loss = torch.nn.L1Loss()



		
		self.cut = 0
		self.latest_desired_goal = self.final_goal
		self.residual_goals_debug = np.array([]).reshape(0,self.state_dim)
		self.latest_achieved = None
		self.sampled_goals = np.array([]).reshape(0,self.state_dim)
		self.debug_trajectories = []
		self.residual_goal_rtg_increase = 1
		self.log_every_n_times =  5
		self.residual_this_episode = False
		self.residuals_till_now = np.array([]).reshape(0,self.state_dim)
		self.video_recorder : VideoRecorder = video_recorder
		self.visualizer = Visualizer(self)	
		self.debug_cluster = []
		# self.subtrajectory_buffer  = TrajectoryBufferer(val_eval_fn=self.value_estimator.get_state_values)
		self.subtrajectory_buffer = Km(val_eval_fn=self.value_estimator.get_state_values,final_goal=self.final_goal)
		self.bayesian_predictor  = BayesianPredictor(input_size= self.state_dim , evaluator=self.subtrajectory_buffer.check_if_close, limits=self.limits, final_goal=self.final_goal)
		self.subtrajectory_buffer.path_evaluator = self.bayesian_predictor.predict
		self.proclaimed_states, self.proclaimed_rtgs = np.array([]).reshape(0,self.state_dim), np.array([]).reshape(0,self.state_dim)
	def reward_to_rtg(self,rewards):
		if isinstance(rewards, np.ndarray):
			rewards_copy =  rewards.copy()
				
			rewards_copy -= rewards[0]
			return rewards_copy[::-1].copy()
		elif isinstance(rewards, torch.Tensor):
			rewards_copy =  rewards.clone()
			rewards_copy -= rewards[0]
			return rewards_copy.flip(0).clone()
		else:
			raise ValueError("Invalid type for rewards")

	@time_decorator
	def update(self, step, episode, achieved_states, actions, qs):
			
		self.step = step
		self.episode = episode
		achieved_states = np.array([self.eval_env.convert_obs_to_dict(achieved_states[i])["achieved_goal"] for i in range(len(achieved_states))])
		actions = np.array(actions)

		rewards, achieved_values, exploration_values, q_values = self.value_estimator.get_state_values(achieved_states, qs)
		rtgs = self.reward_to_rtg(rewards)

		debug_loc_list.append(achieved_states)
		debug_rtg_list.append(rtgs)



		self.latest_achieved 	= np.array([achieved_states])
		self.latest_acts        = np.array([actions])
		self.latest_rtgs 		= np.array([rtgs ])
		self.latest_qs			= q_values

		self.subtrajectory_buffer.add_trajectory(achieved_states)
		self.bayesian_predictor.train(np.concatenate((achieved_states,self.subtrajectory_buffer.centroid_buffer)))
		goals_predicted_debug_np = self.train(achieved_states, rtgs)

		if self.episode % self.log_every_n_times == 0:

				self.visualize_value_heatmaps_for_debug(goals_predicted_debug_np, self.proclaimed_states, self.proclaimed_rtgs)
				self.residual_this_episode = False
				self.debug_trajectories = []


		self.residual_goals_debug = np.array([]).reshape(0,self.state_dim)
		self.debug_traj_lens = []


	@time_decorator
	def train(self, achieved_states, rtgs):
		goals_predicted_debug_np = np.array([]).reshape(0,self.state_dim)
		max_distance = calculate_max_distance(achieved_states)	
		self.residual_goal_length = 0
		for trajectory, rtgs_t, traj_len in self.subtrajectory_buffer.sample():#self.trajectory_reconstructor.get_shortest_jump_tree(achieved_states, top_n=25, eval_fn = self.value_estimator.get_state_values, pick_n= 10):

			trajectory_t 								= torch.tensor(trajectory, device="cuda", dtype=torch.float32).unsqueeze(0)
			rtgs_t 										= torch.tensor(rtgs_t, device="cuda", dtype=torch.float32).unsqueeze(0)
			actions_t 									= torch.zeros((1, trajectory_t.size(1), self.state_dim), device="cuda", dtype=torch.float32)
			timesteps_t 								= torch.arange(trajectory_t.size(1), device="cuda").unsqueeze(0)

			input_achieved_t 							= trajectory_t[:, :-traj_len, :]
			input_actions_t 							= actions_t[:, :-traj_len, :]
			input_rtgs_t 								= rtgs_t[:, :-traj_len]
			input_timesteps_t 							= timesteps_t[:, :-traj_len]
			input_rtgs_t								= input_rtgs_t.unsqueeze(-1)

			generated_states_t, _, generated_rtgs_t, _  = self.generate_next_n_states(input_achieved_t,input_actions_t,input_rtgs_t,input_timesteps_t, n=traj_len ) 
			exploration_loss_t = 0 #self.estimate_novelty_through_embeddings(generated_states_t, traj_len + self.residual_goal_length)
			distances_t = torch.norm(generated_states_t[-traj_len:, 1:, :] - generated_states_t[-traj_len:, :-1, :], dim=-1)
			distance_to_1 = 0.5 - distances_t
			distance_regulation_loss = torch.mean(distance_to_1 ** 2)

			total_val_t, ach_val_t, expl_val_t, q_val_t = self.value_estimator.get_state_values_t(generated_states_t[:,-traj_len:,:])
			rtg_values_t 								= self.reward_to_rtg(total_val_t)

			goal_val_loss 								= self.goal_val_loss(rtg_values_t[:,:], rtgs_t[:,-traj_len:])
			rtg_pred_loss 								= self.rtg_pred_loss(generated_rtgs_t[:,-traj_len:].squeeze(-1), rtg_values_t[:,:])
			_, dtw_dist, smoothness_reg 				= trajectory_similarity_loss(generated_states_t[:,-traj_len:,:].squeeze(0), trajectory_t[0,-traj_len:,:])
			euc_dist  									= euclidean_distance_loss(generated_states_t[:,-traj_len:,:],  trajectory_t[0,-traj_len:,:])
			
			q_gain_rewards_t 							= torch.diff(q_val_t)
			state_val_gain_rewards_t 					= torch.diff(total_val_t)

			total_loss =  distance_regulation_loss* 10 + dtw_dist + rtg_pred_loss + goal_val_loss + euc_dist + torch.mean(state_val_gain_rewards_t)
			self.state_optimizer.zero_grad()
			total_loss.backward(retain_graph=True)
			torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
			self.state_optimizer.step()


			goals_predicted_debug_np = np.vstack((goals_predicted_debug_np, generated_states_t.squeeze(0)[-traj_len:].detach().cpu().numpy()))
			self.debug_trajectories.append(trajectory[-traj_len:])

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
	
	def forward_n_times(self, achieved_states_t, actions_t, rtgs_t, timesteps_t, n= 10):
		if not isinstance(achieved_states_t, torch.Tensor):
			achieved_states_t = torch.tensor(achieved_states_t, device="cuda", dtype=torch.float32)
		if not isinstance(actions_t, torch.Tensor):
			actions_t = torch.tensor(actions_t, device="cuda", dtype=torch.float32)
		if not isinstance(rtgs_t, torch.Tensor):
			rtgs_t = torch.tensor(rtgs_t, device="cuda", dtype=torch.float32)
		if not isinstance(timesteps_t, torch.Tensor):
			timesteps_t = torch.tensor(timesteps_t, device="cuda", dtype=torch.float32)

		init_goal_t = torch.tensor(self.init_goal, device="cuda", dtype=torch.float32).unsqueeze(0)
		init_rtg    = rtgs_t[0,0].unsqueeze(0).unsqueeze(0)
		init_action = torch.tensor([0,0], device="cuda", dtype=torch.float32).unsqueeze(0).unsqueeze(0)
		for _ in range(n):
			predicted_state_mean_t, predicted_state_logvar, predicted_action_t, predicted_rtg_t = self.dt.forward(
				achieved_states_t, actions_t, None, rtgs_t, timesteps_t)
        	
			achieved_states_t = torch.cat((init_goal_t.unsqueeze(0), predicted_state_mean_t), dim=1)
			actions_t = torch.cat((init_action, predicted_action_t), dim=1)
			rtgs_t = torch.cat((init_rtg, predicted_rtg_t), dim=1)
			timesteps_t = torch.concat((timesteps_t, torch.tensor([timesteps_t[0][-1] + 1], device="cuda").unsqueeze(0)), dim = 1)

		return achieved_states_t, actions_t, rtgs_t, timesteps_t
	def get_ordered_k_means_clusterings(self, states, k=4):
		# Perform K-means clustering
		kmeans = KMeans(n_clusters=k).fit(states)
		labels = kmeans.labels_

		# Initialize a list to store the indices of the last occurrence of each cluster
		last_indices = []

		# Find the last occurrence of each cluster label
		for i in range(k):
			indices = np.where(labels == i)[0]  # Find indices of points belonging to cluster i
			if len(indices) > 0:
				last_index = indices[-1]  # Get the last index for cluster i
				last_indices.append(last_index)

		# Sort the last indices and get the corresponding states
		last_indices.sort()
		ordered_states = np.array([states[i] for i in last_indices])

		return ordered_states
	
	def cumulative_distance_sampling(self, states, distance_threshold = 3):
		# Initialize the list of sampled states
		sampled_states = [states[-1]]

		# Iterate over the states
		for i in range(len(states)-2,1, -1):
			# Calculate the distance between the current state and the last sampled state
			distance = np.linalg.norm(states[i] - sampled_states[-1])

			# If the distance exceeds the threshold, sample the state
			if distance >= distance_threshold:
				sampled_states.append(states[i])

		return np.array(sampled_states)[::-1].copy()
	
	def n_checkpoint_sample(self, states, n=2):
		# Step 1: Get intermediate goals using cumulative distance sampling
		intermediate_goals = self.cumulative_distance_sampling(states, distance_threshold=2.5)
		
		# Step 2: Apply the cut to reduce the range
		cut = min(self.cut, len(intermediate_goals) - 1)  # Use self.cut as the cut value

		intermediate_goals = intermediate_goals[cut:]  # Cut from the start

		# Step 3: Calculate the step size for sampling
		idx_distance = max(1, len(intermediate_goals) // n)
		
		# Step 4: Ensure that at least the last element is included
		if n >= len(intermediate_goals):
			sampled_goals = intermediate_goals  # Take all if n is greater than remaining goals
		else:
			# Reverse sampling logic with gradual diminishing
			sampled_goals = [intermediate_goals[-1]]  # Start with the last element
			
			current_idx = len(intermediate_goals) - 1  # Start from the end
			while len(sampled_goals) < n:
				# Gradually decrease the step size
				idx_distance = max(1, current_idx // (n - len(sampled_goals)))
				current_idx = max(0, current_idx - idx_distance)
				sampled_goals.append(intermediate_goals[current_idx])

			# Reverse to maintain correct order
			sampled_goals.reverse()

		return np.array(sampled_goals)

	@time_decorator
	def sample(self, episode_observes = None, episode_acts = None, qs = None):

		if episode_observes is None or qs is None or episode_acts is None:
			with torch.no_grad():

				if self.latest_achieved is None:
					return self.final_goal
				actions_t = torch.zeros((1, 1, self.state_dim), device="cuda", dtype=torch.float32)
				generated_states_t, _,_,_ = 	self.generate_next_n_states(self.latest_achieved[:,:1], actions_t, np.expand_dims(self.latest_rtgs[:,:1],axis=-1) + 0.1, torch.arange(1, device="cuda").unsqueeze(0), n = 100)

				generated_states = generated_states_t.squeeze(0).detach().cpu().numpy()	
				goal = generated_states[-1]
				sampled_states = self.n_checkpoint_sample(generated_states, n=self.n)
				sampled_states = self.subtrajectory_buffer.get_closest_points(sampled_states)
				self.debug_cluster = sampled_states
				goal = sampled_states[0]

				# sampled_states = self.get_ordered_k_means_clusterings(generated_states, k=4)
				# sampled_states =  self.cumulative_distance_sampling(generated_states, distance_threshold=3)
				# self.debug_cluster = sampled_states
				goal = sampled_states[0]
				self.sampled_states = sampled_states[1:]
				self.proclaimed_states = generated_states
				if np.linalg.norm(goal - self.init_goal) < 1 and len(self.sampled_states) > 0:
					goal = sampled_states[0]
					sampled_states = sampled_states[1:]
				self.latest_desired_goal = goal
				self.residual_goals_debug = np.vstack((self.residual_goals_debug, goal))
				self.residuals_till_now = np.vstack((self.residuals_till_now, [goal]))
				self.sampled_goals = np.concatenate((self.sampled_goals, [goal]))

				self.cut = max(self.cut-1, 0)
				if np.linalg.norm(self.final_goal - goal) < 1:
					return self.final_goal
				return goal
		else:

			if len(self.sampled_states) == 0 :
				return None
			self.cut+=1

			goal = self.sampled_states[0]
			self.sampled_states = self.sampled_states[1:]

			# self.residual_this_episode = True
			self.residual_goals_debug = np.vstack((self.residual_goals_debug, goal))
			self.residuals_till_now = np.vstack((self.residuals_till_now, [goal]))
			print(f"Residual goal generated in episode: {self.episode} in the step : {len(episode_observes) }")
			self.latest_desired_goal = goal
			if np.linalg.norm(self.final_goal - goal) < 1:
				return self.final_goal
			return goal 
	


	@time_decorator
	def visualize_value_heatmaps_for_debug(self, goals_predicted_during_training, proclaimed_states, proclaimed_rtgs):
		self.visualizer.visualize_value_heatmaps_for_debug(goals_predicted_during_training, proclaimed_states, proclaimed_rtgs)

