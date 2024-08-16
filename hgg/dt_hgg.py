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
from playground2 import time_decorator
from debug_utils import plot_positions, plot_two_array_positions
from hgg.loss_utils import cubic_spline_interpolation_loss, trajectory_similarity_loss
debug_loc_list = []
debug_rtg_list = []
class DTSampler:    
	def __init__(self, goal_env, goal_eval_env, 				
				agent = None,
				add_noise_to_goal= False, beta = 2, gamma=-1, sigma = 1, device = 'cuda',
				dt : DecisionTransformer = None,  optimizer = None,
				video_recorder : VideoRecorder = None, rescale_rewards = True, env_name = "PointUMaze-v0"
				):
		if env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
			self.final_goal = np.array([0., 8.])
		elif env_name == "PointSpiralMaze-v0":
			self.final_goal = np.array([8., -8.])
		elif env_name in ["PointNMaze-v0"]:
			self.final_goal = np.array([8., 16.])

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
		self.trajectory_reconstructor = TrajectoryReconstructor(merge_when_full=True)

		# Losses
		self.path_similarity_loss = trajectory_similarity_loss
		self.rtg_pred_loss = torch.nn.L1Loss()
		self.goal_val_loss = torch.nn.L1Loss()


		self.limits_map = {"PointUMaze-v0": [[-2,10],[-2,10]], "PointSpiralMaze-v0": [[-10,10],[-10,10]], "PointNMaze-v0": [[-2,10],[-2,18]]}
		self.limits = self.limits_map[self.env_name]
		

		self.latest_desired_goal = self.final_goal
		self.max_rewards_so_far = []
		self.residual_goals_debug = np.array([]).reshape(0,2)
		self.latest_achieved = None
		self.max_achieved_reward = 0
		self.sampled_goals = np.array([]).reshape(0,2)
		self.sampled_states = np.array([]).reshape(0,2)
		self.debug_trajectories = []
		self.debug_traj_lens = []
		self.residual_goal_rtg_increase = 1
		self.log_every_n_times =  1
		self.residual_this_episode = False
		self.residuals_till_now = np.array([]).reshape(0,2)
		self.video_recorder : VideoRecorder = video_recorder
		self.visualizer = Visualizer(self)	
	def reward_to_rtg(self,rewards):
		return rewards - rewards[-1]

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
		self.max_rewards_so_far.append(max(rewards))
		self.sampled_states = np.array([]).reshape(0,2)
		self.max_achieved_reward = max(max(rewards),self.max_achieved_reward)


		goals_predicted_debug_np = self.train(achieved_states, rtgs)


		if self.episode % self.log_every_n_times == 0:
			proclaimed_states, _ , proclaimed_rtgs, _ = self.generate_next_n_states(self.latest_achieved[:,:20], self.latest_acts[:,:20], np.expand_dims(self.latest_rtgs[:,:20],axis=-1), torch.arange(20, device="cuda").unsqueeze(0), n = 90)
			proclaimed_states = proclaimed_states.squeeze(0).detach().cpu().numpy() 
			proclaimed_rtgs = proclaimed_rtgs.squeeze(0).detach().cpu().numpy()
			self.visualize_value_heatmaps_for_debug(goals_predicted_debug_np, proclaimed_states, proclaimed_rtgs)
			self.residual_this_episode = False
			self.debug_trajectories = []


		self.residual_goals_debug = np.array([]).reshape(0,2)
		self.debug_traj_lens = []

	def estimate_novelty_through_embeddings(self, generated_states, len_traj, last_n=10):
		if not isinstance(generated_states, torch.Tensor):
			generated_states = torch.tensor(generated_states, device="cuda", dtype=torch.float32)
		
		# Get the embeddings for the generated states
		embeddings = self.dt.get_state_embeddings(generated_states)
		
		# Split the embeddings into achieved and generated parts using slicing
		achieved_embeddings = embeddings[:, :generated_states.shape[1] - len_traj]
		generated_embeddings = embeddings[:, generated_states.shape[1] - len_traj:]
		
		# Initialize the loss tensor
		loss_t = torch.tensor([0], device="cuda", dtype=torch.float32)
		
		# Iterate over each generated embedding
		for gen_embedding in generated_embeddings[0]:
			# Calculate Euclidean distance with the last n achieved embeddings
			if achieved_embeddings.shape[1] >= last_n:
				last_achieved_embeddings = achieved_embeddings[0, -last_n:]
			else:
				last_achieved_embeddings = achieved_embeddings[0]
			
			# Euclidean distance calculation
			euclidean_dist = torch.norm(last_achieved_embeddings - gen_embedding.unsqueeze(0), dim=1)
			
			# Normalize the Euclidean distance by the square root of the embedding dimension
			normalized_dist = euclidean_dist / torch.sqrt(torch.tensor(last_achieved_embeddings.size(1), device="cuda", dtype=torch.float32))
			
			# Aggregate the normalized distances (sum or mean)
			loss_t += normalized_dist.mean()  # or .sum() depending on your application
			
			# Append the current generated embedding to the achieved embeddings
			achieved_embeddings = torch.cat((achieved_embeddings, gen_embedding.unsqueeze(0).unsqueeze(1)), dim=1)
		
		return loss_t
	

			
			
	def cosine_similarity(self, a, b):
		return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

		
		

	@time_decorator
	def train(self, achieved_states, rtgs):
		goals_predicted_debug_np = np.array([[0,0]])
		max_distance = calculate_max_distance(achieved_states)	
		self.residual_goal_length = 5
		for trajectory, rtgs_t, traj_len in self.trajectory_reconstructor.get_shortest_jump_tree(achieved_states, top_n=30, eval_fn = self.value_estimator.get_state_values, pick_n= 10):

			trajectory_t 								= torch.tensor(trajectory, device="cuda", dtype=torch.float32).unsqueeze(0)
			rtgs_t 										= torch.tensor(rtgs_t, device="cuda", dtype=torch.float32).unsqueeze(0)
			actions_t 									= torch.zeros((1, trajectory_t.size(1), 2), device="cuda", dtype=torch.float32)
			timesteps_t 								= torch.arange(trajectory_t.size(1), device="cuda").unsqueeze(0)

			input_achieved_t 							= trajectory_t[:, :-traj_len, :]
			input_actions_t 							= actions_t[:, :-traj_len, :]
			input_rtgs_t 								= rtgs_t[:, :-traj_len]
			input_timesteps_t 							= timesteps_t[:, :-traj_len]
			input_rtgs_t								= input_rtgs_t.unsqueeze(-1)

			generated_states_t, _, generated_rtgs_t, _  = self.generate_next_n_states(input_achieved_t,input_actions_t,input_rtgs_t,input_timesteps_t, n=traj_len + self.residual_goal_length) 
			exploration_loss_t = 0 #self.estimate_novelty_through_embeddings(generated_states_t, traj_len + self.residual_goal_length)
			distances_t = torch.norm(generated_states_t[-(traj_len + self.residual_goal_length+1):, 1:] - generated_states_t[-(traj_len + self.residual_goal_length+1):, :-1], dim=-1)
			distance_to_1 = distances_t
			distance_regulation_loss = torch.mean(distance_to_1 ** 2)

			generated_states_t, residual_states_t 		= np.split(generated_states_t,[-self.residual_goal_length], axis= 1)
			generated_rtgs_t 							= generated_rtgs_t[:,:-self.residual_goal_length]

			total_val_t, ach_val_t, expl_val_t, q_val_t = self.value_estimator.get_state_values_t(generated_states_t[:,-traj_len-1:,:])
			rtg_values_t 								= self.reward_to_rtg(total_val_t)

			goal_val_loss 								= self.goal_val_loss(rtg_values_t[:,1:], rtgs_t[:,-traj_len:])
			rtg_pred_loss 								= self.rtg_pred_loss(generated_rtgs_t[:,-traj_len:].squeeze(-1), rtg_values_t[:,1:])
			_, dtw_dist, smoothness_reg 				= trajectory_similarity_loss(generated_states_t[:,-traj_len:,:].squeeze(0), trajectory_t[0,-traj_len:,:])

			q_gain_rewards_t 							= torch.diff(q_val_t)
			state_val_gain_rewards_t 					= torch.diff(total_val_t)

			total_loss =  -exploration_loss_t  + 0.4 * dtw_dist + smoothness_reg * 0.05 + rtg_pred_loss + goal_val_loss -  2* torch.mean(q_gain_rewards_t)

			self.state_optimizer.zero_grad()
			total_loss.backward(retain_graph=True)
			torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
			self.state_optimizer.step()


			goals_predicted_debug_np = np.vstack((goals_predicted_debug_np, generated_states_t.squeeze(0)[-traj_len:].detach().cpu().numpy()))
			self.debug_trajectories.append(trajectory[-traj_len:])
			self.debug_traj_lens.append(traj_len)

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

