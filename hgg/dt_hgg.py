from queue import PriorityQueue
import math
import random
import numpy as np
from hgg.gcc_utils import gcc_load_lib, c_double, c_int
import torch
import torch.nn.functional as F
from dt.models.decision_transformer import DecisionTransformer
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from video import VideoRecorder
import heapq
from hgg.utils import MoveOntheLastPartLoss
from scipy.ndimage import convolve
from hgg.utils import BiggerNN
import torch.optim as optim
from hgg.utils import generate_random_samples
from hgg.utils import euclid_distance
from hgg.dt_hgg_visualizer import Visualizer
from hgg.utils import trajectory_similarity_loss
from hgg.utils import find_diminishing_trajectories
from hgg.value_estimator import ValueEstimator
from hgg.states_buffer import StatesBuffer
from hgg.trajectory_buffer import TrajectoryBuffer
from hgg.trajectory_sampler import identify_good_parts
from hgg.trajectory_sampler import select_segment_indices
from hgg.sampler_utils import shortest_path_trajectory, get_shortest_path_trajectories
from video import VideoRecorder
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
		self.loss_fn = torch.nn.MSELoss()

		self.dim = np.prod(self.env.convert_obs_to_dict(self.env.reset())['achieved_goal'].shape)
		self.video_recorder : VideoRecorder = video_recorder
		
		self.init_goal = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['achieved_goal'].copy()
		
				
		self.final_goal = np.array([0,8]) #TODO make this parametrized
		self.latest_achieved = None
		self.dt = dt
		self.loss_fn = loss_fn
		self.max_achieved_reward = 0
		self.return_to_add = 0.05
		self.discount_rate = 1# 0.99 #TODO add this as init value
		self.limits = [[-2,10],[-2,10], [0,0]] # TODO make this generic
		self.latest_desired_goal = self.final_goal
		self.max_rewards_so_far = []
		self.residual_goals_debug = []
		self.number_of_trajectories_per_episode = 2

		self.negatives_buffer = StatesBuffer()
		self.bnn_model = BiggerNN(2,1).to(device)
		self.bnn_region_optimizer = optim.Adam(self.bnn_model.parameters(), lr=0.001)
		self.bnn_trajectory_optimizer = optim.Adam(self.bnn_model.parameters(), lr=0.001)
		self.negatives_buffer.fill_list_with_states_near_a_point(self.init_goal)
		self.positives_buffer = StatesBuffer(200)
		self.positives_buffer.fill_list_with_random_around_maze(self.limits, 100)
		self.sampled_goals = np.array([[0,0]])
		self.trajectory_buffer = TrajectoryBuffer(100)

		self.value_estimator = ValueEstimator(self.agent, self.eval_env, self.final_goal, self.num_seed_steps, self.init_goal, gamma, beta, sigma )
		self.visualizer = Visualizer(self)	
	def reward_to_rtg(self,rewards):
		return rewards - rewards[-1]



	def get_positives(self,sample_size = 32):
		positives = np.tile(np.array([self.final_goal], dtype = np.float64), (sample_size,1))
		return positives
	

	def train_bnn(self, achieved_goals_negatives):
		negatives = self.negatives_buffer.sample(32)
		negatives = np.concatenate((negatives, achieved_goals_negatives[-32:]))
		positives = self.positives_buffer.sample(64)

		neg_labels = np.zeros(len(negatives))
		pos_labels = np.ones(len(positives))

		x_train = np.concatenate((negatives, positives), axis=0)
		y_train = np.concatenate((neg_labels, pos_labels), axis=0)
		
		x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
		y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
		
		
		self.bnn_model.train()
		self.bnn_region_optimizer.zero_grad()
		outputs = self.bnn_model(x_train).squeeze()
		outputs = torch.sigmoid(outputs)

		loss = torch.nn.BCEWithLogitsLoss()(outputs, y_train)
		loss.backward()
		self.bnn_region_optimizer.step()


	def update(self, step, episode, achieved_goals, actions, qs):
			
		self.step = step
		self.episode = episode

		achieved_goals_states = np.array([self.eval_env.convert_obs_to_dict(achieved_goals[i])["achieved_goal"] for i in range(len(achieved_goals))])
		actions = np.array(actions)

		if not len(qs):
			qs = np.zeros((100,2))
		rewards, achieved_values, exploration_values, q_values = self.value_estimator.get_state_values(achieved_goals_states, qs)
		rtgs = self.reward_to_rtg(rewards)
		debug_loc_list.append(achieved_goals_states)
		debug_rtg_list.append(rtgs)

		start_ends = find_diminishing_trajectories( rtgs, dynamic_threshold=True)
		start_ends = select_segment_indices(rtgs, 10, 0, 0.01, 2)
		self.start_ends = start_ends
		# trajectories = get_shortest_path_trajectories(achieved_goals_states, rtgs, 5)
		
		self.trajectory_buffer.add_trajectory(achieved_goals_states, actions, start_ends)
		start_end_index = self.trajectory_buffer.sample(10)


		self.latest_achieved 	= np.array([achieved_goals_states])
		self.latest_acts        = actions
		self.latest_rtgs 		= np.array([rtgs ])


		self.max_achieved_reward = max(max(rewards),self.max_achieved_reward)
		self.latest_qs = q_values
		

		self.negatives_buffer.insert_trajectory(achieved_goals_states)
		self.positives_buffer.remove_states_near_other_sample(self.negatives_buffer.sample(len(self.negatives_buffer)//2), 0.5)
		if len(self.positives_buffer) < 64 : #TODO make it class variable
			self.positives_buffer.fill_list_with_random_around_maze(self.limits, 64 - len(self.positives_buffer)) # sample the same thing
		
	
		loss = self.train(achieved_goals_states, actions, rtgs, start_end_index) #visualize this

		self.max_rewards_so_far.append(max(rewards))

		self.residual_goals_debug = []

	def train(self, achieved_goals, actions, rtgs, start_end):
		goals_predicted_debug_np = np.array([[0,0]])

		# actions = torch.zeros((1, achieved_goals_t.size(1), 2), device="cuda", dtype=torch.float32)
		timesteps = torch.arange(100, device="cuda").unsqueeze(0)
		

		for index, start, end in start_end:
			achieved_goals = self.trajectory_buffer.get_trajectory_by_index(index)
			actions = self.trajectory_buffer.get_acts_by_index(index)
			
			achieved_goals_t = torch.tensor([achieved_goals], device="cuda", dtype=torch.float32)
			actions_t = torch.tensor([actions], device = "cuda", dtype = torch.float32)
			rtgs_t,_,_,_ = self.value_estimator.get_state_values_t(achieved_goals_t) #TODO only get the values for the part of the trajectory 

			temp_achieved = achieved_goals_t[:, :start].clone()
			temp_actions = actions_t[:, :start].clone()
			temp_timesteps = timesteps[:, :start].clone()

			temp_rtg = rtgs_t[:, :start].clone()
			temp_rtg -= rtgs_t[0,end]
			temp_rtg = temp_rtg.unsqueeze(-1)

			generated_states, generated_actions, generated_rtgs, generated_timesteps = self.generate_next_n_states(
				temp_achieved, temp_actions, temp_rtg, temp_timesteps, n=end-start)

			goals_predicted_debug_np = np.vstack((goals_predicted_debug_np, generated_states.squeeze(0)[-end+start:].detach().cpu().numpy()))
			

			expected_val = temp_rtg[0, 0, 0]
			state_values, achieved_values_t, exploration_values_t, q_values_t = self.value_estimator.get_state_values_t(generated_states[:,start-end:,:])

			goal_val_loss = torch.nn.L1Loss()(state_values, rtgs_t.squeeze(0)[start: end])
			rtg_pred_loss = torch.nn.L1Loss()(generated_rtgs[:,start - end:], state_values)
			similarity_loss, dtw_distance, euclidean_distance, smoothness_reg =  trajectory_similarity_loss(generated_states[:,start-end:,:], achieved_goals_t[0,start:end])
			
			state_reward = state_values.sum()

			# Combine losses
			total_loss =   dtw_distance + smoothness_reg -  state_reward + rtg_pred_loss + goal_val_loss
			self.state_optimizer.zero_grad()
			total_loss.backward(retain_graph=True)
			torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
			self.state_optimizer.step()

		self.visualize_value_heatmaps_for_debug(goals_predicted_debug_np)
		
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
	
	def sample(self, episode_observes = None, episode_acts = None, qs = None):
		if episode_observes is None or qs is None or episode_acts is None:
			if self.latest_achieved is None:
				return np.array([0,8])
			# goal_t =  self.generate_goal(self.latest_achieved,self.latest_rtgs + self.return_to_add)
			actions_t = torch.tensor(self.latest_acts, device="cuda", dtype=torch.float32).unsqueeze(0)
			rtgs_t = torch.tensor(self.latest_rtgs, device="cuda", dtype=torch.float32).unsqueeze(2)
			achieved_goals_states_t = torch.tensor(self.latest_achieved, device="cuda", dtype=torch.float32)
			generated_states, _,_,_ = 	self.generate_next_n_states(achieved_goals_states_t, actions_t, rtgs_t, torch.arange(achieved_goals_states_t.shape[1], device="cuda").unsqueeze(0), n = 10)
			goal_t = generated_states[0,-1]
			# trajectories, rtgs = self.best_trajectories.get_elements()
			# goal_t = self.generate_goal(episode_observes, rtgs[0] + self.return_to_add )
			goal =  goal_t.detach().cpu().numpy()
			self.latest_desired_goal = goal
			self.sampled_goals = np.concatenate((self.sampled_goals, [goal]))
			return goal
		else:
			achieved_goals_states = np.array([self.eval_env.convert_obs_to_dict(episode_observes[i])["achieved_goal"] for i in range(len(episode_observes))])
			actions = np.array(episode_acts)

			rewards, achieved_values, exploration_values, q_values = self.value_estimator.get_state_values(achieved_goals_states, qs)
			rtgs = self.reward_to_rtg(rewards)
   
			achieved_goals_states_t = torch.tensor([achieved_goals_states], device="cuda", dtype=torch.float32)
			rtgs_t = torch.tensor([rtgs], device="cuda", dtype=torch.float32).unsqueeze(2)
			actions_t = torch.tensor([actions], device="cuda", dtype=torch.float32)#.unsqueeze(0)

			generated_states, _,_,_ = self.generate_next_n_states(achieved_goals_states_t, actions_t, rtgs_t, torch.arange(achieved_goals_states_t.shape[1], device="cuda").unsqueeze(0), n = 10)
			goal_t = generated_states[0,-1]
			# goal_t = self.generate_goal(achieved_goals_states_t, rtgs_t + self.return_to_add)
			goal =  goal_t.detach().cpu().numpy()
   
			self.latest_desired_goal = goal
			self.residual_goals_debug.append(goal)

			return goal


	def generate_goal(self, achieved_goals, rtgs):
		actions = torch.zeros((1, achieved_goals.shape[0], 2), device="cuda", dtype=torch.float32)
		timesteps = torch.arange(achieved_goals.shape[0], device="cuda").unsqueeze(0)  # Adding batch dimension
		achieved_goals = torch.tensor(achieved_goals, device="cuda", dtype=torch.float32)

		rtgs = torch.tensor(rtgs, device = "cuda", dtype = torch.float32).unsqueeze(0)
		while True:
			desired_goal = self.dt.get_state(achieved_goals, actions, None, rtgs, timesteps)[0] # TODO set this right this is just patchwork
			loss = 0
			if (desired_goal[0] < self.limits[0][0]):
				loss += (self.limits[0][0] - desired_goal[0])**2
			if (desired_goal[0] > self.limits[0][1]):
				loss += (desired_goal[0] - self.limits[0][1])**2
			if (desired_goal[1] < self.limits[1][0]):
				loss += (self.limits[1][0] - desired_goal[1])**2
			if (desired_goal[1] > self.limits[1][1]):
				loss += (desired_goal[1] - self.limits[1][1])**2
			if loss > 0:
				self.state_optimizer.zero_grad()
				loss.backward(retain_graph=True)  # retain the computation graph
				torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
				self.state_optimizer.step()
			else:
				self.positives_buffer.insert_state(desired_goal.detach().cpu().numpy())
				return desired_goal.detach()  # return the desired goal if within limits
			
	def get_state_values(self, achieved_goals):
		achieved_values_t = self.value_estimator.generate_achieved_values_t(self.init_goal, achieved_goals[0])
		exploration_values_t = torch.tensor([], device="cuda", dtype=torch.float32, requires_grad=True)
		q_values_t = torch.tensor([], device="cuda", dtype=torch.float32, requires_grad=True)

		init_goal_t = torch.tensor(self.init_goal, device="cuda", dtype=torch.float32)        

		for i, state in enumerate(achieved_goals[0]):
			exploration_value_t = self.value_estimator.calculate_exploration_value(init_goal_t, state)
			exploration_value_t = exploration_value_t.unsqueeze(0)  # Add an extra dimension
			exploration_values_t = torch.cat((exploration_values_t, exploration_value_t))

			q1_t, q2_t = self.value_estimator.get_q_values(state)
			q_val_t = torch.min(q1_t, q2_t)
			q_val_t = q_val_t.unsqueeze(0)  # Add an extra dimension
			q_values_t = torch.cat((q_values_t, q_val_t))
		q_values_t = q_values_t.squeeze(1)
		state_values_t = self.gamma * achieved_values_t + self.beta * q_values_t + self.sigma * exploration_values_t
		return state_values_t, achieved_values_t, exploration_values_t, q_values_t


	def get_probabilities_over_trajectory(self, trajectory):
		probabilities = np.array([])
		for achieved_goal in trajectory:
			output = self.bnn_model(torch.tensor(achieved_goal, device = "cuda", dtype = torch.float32))
			probability = torch.sigmoid(output)
			probabilities = np.append(probabilities,probability.unsqueeze(0).detach().cpu().numpy())
		return probabilities

	def visualize_value_heatmaps_for_debug(self, goals_predicted_during_training):
		self.visualizer.visualize_value_heatmaps_for_debug(goals_predicted_during_training)

