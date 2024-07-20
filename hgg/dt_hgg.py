from queue import PriorityQueue
import math
import random
import numpy as np
from hgg.gcc_utils import gcc_load_lib, c_double, c_int
import torch
import torch.nn.functional as F
import time
import copy
import itertools
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
class StatesBuffer:
	def __init__(self, max_size=5000):
		self.__list = np.array([[0, 0]])
		self.max_size = max_size

	def insert_trajectory(self, trajectory):
		self.__list = np.vstack((trajectory, self.__list))[:self.max_size]

	def insert_state(self, state):
		self.__list = np.vstack((self.__list , np.array([state])))

	def fill_list_with_states_near_a_point(self, init_state):
		temp_list = np.tile(np.array([init_state], dtype = np.float64), (self.max_size,1))
		temp_list += np.random.normal(loc=np.zeros_like(self.__list), scale=0.5*np.ones_like(self.__list))
		self.__list = np.vstack((self.__list , temp_list))

	def fill_list_with_random_around_maze(self, limits, sample_size ):
		temp_list = generate_random_samples([limits[0][0], limits[1][0]], [limits[0][1], limits[1][1]], (2,) , sample_size)
		self.__list = np.vstack((self.__list, temp_list))
		pass #debug

	def sample(self, batch_size=64):
		sampled_elements = self.__list[np.random.choice(self.__list.shape[0], batch_size, replace=False)]
		return sampled_elements
	
	def remove_states_near_the_trajectory(self, trajectory, threshold):
		for state in trajectory:
			self.remove_states_near_the_state(state, threshold)
	def remove_states_near_other_sample(self, sample, threshold):
		for state in sample:
			self.remove_states_near_the_state(state, threshold)
	def remove_states_near_the_state(self, state, threshold):
		delete_indices = []
		for i, s in enumerate(self.__list): #TODO rename
			if euclid_distance(s,state) < threshold:
				delete_indices.append(i)
		self.__list = np.delete(self.__list, delete_indices, axis = 0)

	def __len__(self):
		return len(self.__list)
		



def rescale_array(tensor, old_min, old_max, new_min =-1, new_max = 1):
    # rescaled_tensor = (tensor - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    # return rescaled_tensor
	return tensor

def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a - goal_b, ord=2)
def goal_concat(obs, goal):
	if type(obs) == torch.Tensor:
		if type(goal) == torch.Tensor:
			return torch.concat([obs, goal])
		else:
			goal = torch.tensor(goal, device = "cuda", dtype = torch.float32)
			return torch.concat([obs, goal])
	elif type(goal) == torch.Tensor:
		obs = torch.tensor(obs, device = "cuda", dtype = torch.float32)
		return torch.concat([obs, goal])
	else:
		return np.concatenate([obs, goal], axis=0)

def discount_cumsum(x, gamma):
	discount_cumsum = np.zeros_like(x)
	discount_cumsum[-1] = x[-1]
	for t in reversed(range(x.shape[0]-1)):
		discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
	return discount_cumsum

class DTSampler:    
	def __init__(self, goal_env, goal_eval_env, 				
				agent = None,
				add_noise_to_goal= False, beta = 1/20, gamma=1, device = 'cuda', critic = None, 
				dt : DecisionTransformer = None, rtg_optimizer = None, state_optimizer = None,
				loss_fn  = torch.nn.MSELoss(),
				video_recorder : VideoRecorder = None
				):
		# Assume goal env
		self.env = goal_env
		self.eval_env = goal_eval_env
		
		self.add_noise_to_goal = add_noise_to_goal
		self.agent = agent
		self.rtg_optimizer = rtg_optimizer
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
		self.goal_distance = goal_distance
		self.video_recorder = video_recorder
		self.visualizer = Visualizer(self)	
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
		self.dumb_prevention_loss = MoveOntheLastPartLoss(threshold=4)

		self.negatives_buffer = StatesBuffer()
		self.bnn_model = BiggerNN(2,1).to(device)
		self.bnn_region_optimizer = optim.Adam(self.bnn_model.parameters(), lr=0.001)
		self.bnn_trajectory_optimizer = optim.Adam(self.bnn_model.parameters(), lr=0.001)
		self.negatives_buffer.fill_list_with_states_near_a_point(self.init_goal)
		self.positives_buffer = StatesBuffer(200)
		self.positives_buffer.fill_list_with_random_around_maze(self.limits, 100)
		self.sampled_goals = np.array([[0,0]])
		pass # debug
	def reward_to_rtg(self,rewards):
		return rewards - rewards[-1]


	def generate_achieved_values(self,init_state,achieved_goals):
		# maybe for all timesteps in an episode
		init_to_goal_pair	= 	[goal_concat(init_state, achieved_goals[j]) for  j in range(len(achieved_goals))] # list of [dim] (len = ts)
		if type(init_to_goal_pair) is not torch.Tensor:
			init_to_goal_pair = torch.tensor(init_to_goal_pair, device = "cuda", dtype = torch.float32)
		goal_to_final_pair	= 	[goal_concat(achieved_goals[j], self.final_goal) for  j in range(len(achieved_goals))] # list of [dim] (len = ts) 																													# merge of achieved_pool and achieved_pool_init_state to draw trajectory
		if type(goal_to_final_pair) is not torch.Tensor:
			goal_to_final_pair = torch.tensor(goal_to_final_pair, device = "cuda", dtype = torch.float32)
		with torch.no_grad(): ## when using no_grad, no gradients will be calculated or stored for operations on tensors, which can reduce memory usage and speed up computations							
			values = -self.agent.aim_discriminator(init_to_goal_pair).detach().cpu().numpy()[:, 0] # TODO discover inside aim_discriminator,
																							# 	* what kind of inputs it does require
																							# 	* value can be interpreted as a measure of how desirable or advantageous the current state is from the perspective of achieving the final goal
			# values += self.agent.aim_discriminator(goal_to_final_pair_t).detach().cpu().numpy()[:, 0] 
				# value = np.clip(value, -1.0/(1.0-self.gamma), 0)
		return values
	def generate_achieved_values_t(self,init_state,achieved_goals):
		# maybe for all timesteps in an episode
		# Create a list of tensors by concatenating init_state and each achieved_goal
		goal_pairs = [goal_concat(init_state, achieved_goals[j]) for j in range(len(achieved_goals))]

		# Stack the list of tensors into a single tensor
		init_to_goal_pair = torch.stack(goal_pairs).to(device='cuda', dtype=torch.float32)

		if type(init_to_goal_pair)==torch.Tensor:
			init_to_goal_pair = torch.tensor(init_to_goal_pair, device = "cuda", dtype = torch.float32)

		goal_to_final_pair	= 	[goal_concat(achieved_goals[j], self.final_goal) for  j in range(len(achieved_goals))] # list of [dim] (len = ts) 																													# merge of achieved_pool and achieved_pool_init_state to draw trajectory
		if type(goal_to_final_pair)==torch.Tensor:
			goal_to_final_pair = torch.tensor(goal_to_final_pair, device = "cuda", dtype = torch.float32)

		values = -self.agent.aim_discriminator(init_to_goal_pair)[:, 0] # TODO discover inside aim_discriminator,
																							# 	* what kind of inputs it does require
																							# 	* value can be interpreted as a measure of how desirable or advantageous the current state is from the perspective of achieving the final goal
			# values += self.agent.aim_discriminator(goal_to_final_pair_t).detach().cpu().numpy()[:, 0] 
				# value = np.clip(value, -1.0/(1.0-self.gamma), 0)
		return values
				
	def get_q_values(self,goal_t):
		goal = goal_t.detach().cpu().numpy()
		obs_np = self.eval_env.reset()
		self.eval_env.wrapped_env.set_xy(goal)
		pos_xy_np = self.eval_env.wrapped_env.get_xy()
		obs_np[:2] = pos_xy_np[:]
		obs_np[6:8] = pos_xy_np[:]
		with torch.no_grad():
			obs_t = torch.tensor(obs_np, device = "cuda", dtype = torch.float32)
			dist = self.agent.actor(obs_t)
			action = dist.rsample()

			q1, q2 = self.agent.critic(obs_t.unsqueeze(0), action.unsqueeze(0))

			q_min = torch.min(q1,q2)
			# q_mean = torch.mean(q1+q2)
			# q_max = torch.max(q1,q2)
		if  self.step > self.num_seed_steps:
			return q1,q2
		else: 
			return torch.tensor([0], device = "cuda", dtype=torch.float32), torch.tensor([0], device = "cuda", dtype=torch.float32) #TODO zero is no go, either average the other rewards or pull the calculation of q_vals to start
	def get_dir_of_agent(self, goal_t):
		goal = goal_t.detach().cpu().numpy()
	def calculate_exploration_value(self, init_pos, curr_pos):
		epsilon = 1e-10  # Small value to prevent division by zero
		if type(init_pos) == torch.Tensor:
			numerator = torch.linalg.norm(curr_pos - init_pos)
			denominator = torch.linalg.norm(torch.tensor(self.final_goal, device= "cuda") - curr_pos) + epsilon
			value = torch.log(numerator + epsilon) - torch.log(denominator)

			# value = numerator / denominator
			return value			
		else:
			numerator = np.linalg.norm(curr_pos - init_pos)
			denominator = np.linalg.norm(self.final_goal - curr_pos) + epsilon
			value = np.log(numerator + epsilon) - np.log(denominator)

			# value = numerator / denominator
			return value
	 

	def shorten_trajectory(self, achieved_goals, rtgs):
		achieved_goals = achieved_goals[::3]# list of reduced ts		)
		rtgs = rtgs[::3]
  
		return achieved_goals, rtgs

	def get_max_min_rewards(self, init_state):
		q1_t, q2_t = self.get_q_values(torch.tensor(self.final_goal, device = "cuda", dtype = torch.float32))
		q_val = np.min([q1_t.detach().cpu().numpy(), q2_t.detach().cpu().numpy()])
		aim_val = self.generate_achieved_values(init_state, [self.final_goal])[0]
		expl_val = self.calculate_exploration_value(init_state, self.final_goal)

		max_aim 	= aim_val
		max_expl	= expl_val
		max_q		= q_val
  
		q1_t, q2_t = self.get_q_values(torch.tensor(init_state, device = "cuda", dtype = torch.float32))
		q_val = np.min([q1_t.detach().cpu().numpy(), q2_t.detach().cpu().numpy()])
		aim_val = self.generate_achieved_values(init_state, [init_state])[0]
		expl_val = self.calculate_exploration_value(init_state, init_state)
  
		min_aim 	= aim_val
		min_expl	= expl_val
		min_q		= q_val

		val_dict = {'max_aim' : max_aim,
              		'min_aim' : min_aim,
              		'max_expl' : max_expl,
              		'min_expl' : min_expl,
              		'max_q' : max_q + 1e-10 ,
              		'min_q' : min_q - 1e-10 ,
                }
		return val_dict


	def get_rescaled_rewards(self,achieved_goals, qs):
		achieved_values= self.generate_achieved_values(self.init_goal,achieved_goals)
		exploration_values = np.array([self.calculate_exploration_value(self.init_goal,achieved_goals[i]) for i in range(len(achieved_goals))])

		val_dict = self.get_max_min_rewards(achieved_goals[0])
		if qs is None:
			qs = np.array([])
			for pos in achieved_goals:
				q1_t, q2_t = self.get_q_values(torch.tensor(pos, device="cuda", dtype=torch.float32))

				q_min_t = torch.min(q1_t,q2_t)
				q_min = q_min_t.detach().cpu().numpy()
				qs = np.append(qs,q_min)
		else:
			qs = np.min(qs, axis= 1)

		# achieved_values		=	rescale_array(achieved_values, val_dict['min_aim'], val_dict["max_aim"])
		exploration_values	=	rescale_array(exploration_values, val_dict['min_expl'], val_dict["max_expl"])
		q_values			=	rescale_array(qs, val_dict['min_q'], val_dict["max_q"])
  
		return achieved_values, exploration_values, q_values, val_dict

	def get_positives(self,sample_size = 32):
		positives = np.tile(np.array([self.final_goal], dtype = np.float64), (sample_size,1))
		# positives += np.random.normal(loc=np.zeros_like(positives), scale=0.5*np.ones_like(positives))
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
		
		
		#achieved pool has the whole trajectory throughtout the episode, while achieved_pool_init_state has the initial state where it started the episode
		# achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad() # dont care about pad, it receives the stored achieved trajectories
		if not len(qs):
			qs = np.zeros((100,2))
		self.step = step
		self.episode = episode
		# achieved_goals_states = []
		# # achieved_goals_velocity = []
		# for state in achieved_goals:
		# 	obs_dict = self.eval_env.convert_obs_to_dict(state)
		# 	achieved_goals_states.append(obs_dict["achieved_goal"])
		# 	# achieved_goals_velocity.append(obs_dict["observation"][-3:-1])
		# achieved_goals_states = np.array(achieved_goals_states)
		# # achieved_goals_velocity = np.array(achieved_goals_velocity)
		achieved_goals_states = np.array([self.eval_env.convert_obs_to_dict(achieved_goals[i])["achieved_goal"] for i in range(len(achieved_goals))])
		actions = np.array(actions)

		achieved_values, exploration_values, q_values, min_max_val_dict = self.get_rescaled_rewards(achieved_goals_states, qs)
		rewards = self.gamma * achieved_values + self.beta * q_values + self.sigma * exploration_values# either get the qs earlier than 2000 steps or remove gamma limitation before then

		rtgs = self.reward_to_rtg(rewards)

		starting_indexes = find_diminishing_trajectories(rtgs)
		self.latest_achieved 	= np.array([achieved_goals_states])
		self.latest_rtgs 		= np.array([rtgs ])

		self.negatives_buffer.insert_trajectory(achieved_goals_states)

		self.max_achieved_reward = max(max(rewards),self.max_achieved_reward)
		self.latest_qs = q_values
		
		self.positives_buffer.remove_states_near_other_sample(self.negatives_buffer.sample(len(self.negatives_buffer)//2), 0.5)
		if len(self.positives_buffer) < 64 : #TODO make it class variable
			self.positives_buffer.fill_list_with_random_around_maze(self.limits, 64 - len(self.positives_buffer)) # sample the same thing
		
		# self.train_bnn(achieved_goals_states[-64:])
		# self.train_single_trajectory(achieved_goals_states, actions, rtgs)
		loss = self.train_with_increasing_states(achieved_goals_states, actions, rtgs, starting_indexes)
		self.max_rewards_so_far.append(max(rewards))
		self.residual_goals_debug = []
	def generate_next_n_states(self, achieved_states_t, actions_t, rtgs_t, timesteps_t, n=10):
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
		
	def get_state_values(self, achieved_goals):
		achieved_values_t = self.generate_achieved_values_t(self.init_goal, achieved_goals[0])
		exploration_values_t = torch.tensor([], device="cuda", dtype=torch.float32, requires_grad=True)
		q_values_t = torch.tensor([], device="cuda", dtype=torch.float32, requires_grad=True)

		init_goal_t = torch.tensor(self.init_goal, device="cuda", dtype=torch.float32)        

		for i, state in enumerate(achieved_goals[0]):
			exploration_value_t = self.calculate_exploration_value(init_goal_t, state)
			exploration_value_t = exploration_value_t.unsqueeze(0)  # Add an extra dimension
			exploration_values_t = torch.cat((exploration_values_t, exploration_value_t))

			q1_t, q2_t = self.get_q_values(state)
			q_val_t = torch.min(q1_t, q2_t)
			q_val_t = q_val_t.unsqueeze(0)  # Add an extra dimension
			q_values_t = torch.cat((q_values_t, q_val_t))
		q_values_t = q_values_t.squeeze(1)
		state_values = self.gamma * achieved_values_t + self.beta * q_values_t + self.sigma * exploration_values_t
		return state_values, achieved_values_t, exploration_values_t, q_values_t


	def train_with_increasing_states(self, achieved_goals, actions, rtgs, starting_indexes):
		goals_predicted_debug_np = np.array([[0,0]])
		
		achieved_goals_t = torch.tensor([achieved_goals], device="cuda", dtype=torch.float32)
		actions_t = torch.tensor([actions], device = "cuda", dtype = torch.float32)
		rtgs_t = torch.tensor([rtgs], device="cuda", dtype=torch.float32)

		# actions = torch.zeros((1, achieved_goals_t.size(1), 2), device="cuda", dtype=torch.float32)
		timesteps = torch.arange(achieved_goals_t.size(1), device="cuda").unsqueeze(0)


		for starting_index in starting_indexes:
			temp_achieved = achieved_goals_t[:, :starting_index].clone()
			temp_actions = actions_t[:, :starting_index].clone()
			temp_timesteps = timesteps[:, :starting_index].clone()

			temp_rtg = rtgs_t[:, :starting_index].clone()
			temp_rtg -= rtgs_t[0, starting_index + 1]
			temp_rtg = temp_rtg.unsqueeze(-1)

			generated_states, generated_actions, generated_rtgs, generated_timesteps = self.generate_next_n_states(
				temp_achieved, temp_actions, temp_rtg, temp_timesteps, n=10)

			goals_predicted_debug_np = np.vstack((goals_predicted_debug_np, generated_states.squeeze(0)[-10:].detach().cpu().numpy()))
			

			if i < achieved_goals_t.shape[1] - 1:
				expected_val = temp_rtg[0, 0, 0]
				state_values, achieved_values_t, explroation_values_t, q_values_t = self.get_state_values(generated_states[:,-10:,:])

				goal_val_loss = torch.nn.L1Loss()(state_values, rtgs_t.squeeze(0)[-10:])
				rtg_pred_loss = torch.nn.L1Loss()(generated_rtgs[:,-10:], state_values)
				similarity_loss, dtw_distance, euclidean_distance, smoothness_reg =  trajectory_similarity_loss(generated_states[:,-10:,:], achieved_goals_t[0,starting_index:starting_index+10])
				
				state_reward = state_values.sum()

				# Combine losses
				total_loss =   dtw_distance * 0.5 + smoothness_reg * 0.3

				self.state_optimizer.zero_grad()
				total_loss.backward(retain_graph=True)
				torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
				self.state_optimizer.step()

		self.visualize_value_heatmaps_for_debug(goals_predicted_debug_np)

		return total_loss.item()
	
	def calculate_information_gain(self, state):
		self.bnn_model.eval()
		if type(state) is torch.Tensor:

			state_t = state
			with torch.no_grad():
				outputs = [self.bnn_model(state_t) for _ in range(10)]
				uncertainty_t = torch.stack(outputs).std(0).mean()
			return uncertainty_t
		else:
			
			state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

			with torch.no_grad():
				outputs = [self.bnn_model(state_t) for _ in range(10)]
				uncertainty_t = torch.stack(outputs).std(0).mean().item()
			return uncertainty_t

	def train_single_trajectory(self, achieved_goals, actions, rtgs):
		goals_predicted_debug = []
		
		achieved_goals_t = torch.tensor([achieved_goals], device="cuda", dtype=torch.float32)
		actions_t = torch.tensor([actions], device = "cuda", dtype = torch.float32)
		rtgs_t = torch.tensor([rtgs], device="cuda", dtype=torch.float32)

		# actions = torch.zeros((1, achieved_goals_t.size(1), 2), device="cuda", dtype=torch.float32)
		timesteps = torch.arange(achieved_goals_t.size(1), device="cuda").unsqueeze(0)


		for i in range(1, achieved_goals_t.shape[1] - 1):
			temp_achieved = achieved_goals_t[:, :i]
			temp_actions = actions_t[:, :i]
			temp_timesteps = timesteps[:, :i]

			temp_rtg = rtgs_t[:, :i].clone()
			temp_rtg -= rtgs_t[0, i + 1]
			temp_rtg = temp_rtg.unsqueeze(-1)

			generated_states, generated_actions, generated_rtgs, generated_timesteps = self.generate_next_n_states(
				temp_achieved, temp_actions, temp_rtg, temp_timesteps, n=10)
			predicted_goal_mean_t, predicted_goal_logvar_t, predicted_action_t ,predicted_return_t = self.dt.forward(
				temp_achieved, temp_actions, None, temp_rtg, temp_timesteps)
			predicted_goal_mean_t = predicted_goal_mean_t[0, -1]
			predicted_goal_logvar_t = predicted_goal_logvar_t[0, -1]
			predicted_goal_std = torch.exp(0.5 * predicted_goal_logvar_t)
			predicted_goal_distribution = torch.distributions.Normal(predicted_goal_mean_t, predicted_goal_std)
			predicted_return_t = predicted_return_t[0,-1]
			predicted_goal_t = predicted_goal_distribution.rsample()
			goals_predicted_debug.append(predicted_goal_t.detach().cpu().numpy())

			if i < achieved_goals_t.shape[1] - 1:
				expected_val = temp_rtg[0, 0, 0]

				init_goal_pair = goal_concat_t(achieved_goals_t[0, 0], predicted_goal_t)

				aim_val_t = self.agent.aim_discriminator(init_goal_pair)
				q1_t, q2_t = self.get_q_values(predicted_goal_t)
				q_val_t = torch.min(q1_t, q2_t)
				exploration_val = self.calculate_exploration_value(achieved_goals_t[0], predicted_goal_t)



				goal_val_t = self.gamma * aim_val_t + q_val_t * self.beta + self.sigma * exploration_val
				best_state_index = torch.argmin(temp_rtg[0])

				current_state = achieved_goals_t[0, best_state_index]

				goal_val_loss = torch.nn.L1Loss()(goal_val_t, expected_val.unsqueeze(0))
				rtg_pred_loss = torch.nn.L1Loss()(predicted_return_t, goal_val_t)
				# Calculate Information Gain Reward
				# outputs = self.bnn_model(predicted_goal_t.unsqueeze(0))  # Get probability prediction
				# probability_pred = torch.sigmoid(outputs)
				# uncertainty_loss = torch.abs(probability_pred - 0.0).mean()  # Treat uncertainty as the difference from 0.5
				euclid_distance_loss = torch.sqrt((predicted_goal_t - achieved_goals_t[0,i])**2).sum()
				action_loss = torch.sqrt((predicted_action_t - actions_t[0,i])**2).sum()
				# info_gain_loss = torch.tensor(info_gain_rewards.mean(), dtype=torch.float32, device=self.device)

				# Combine losses
				total_loss = goal_val_loss + rtg_pred_loss + euclid_distance_loss + action_loss# + uncertainty_loss #+ euclid_distance_loss+ action_loss#+ predicted_return_t#- q_val_t

				self.state_optimizer.zero_grad()
				total_loss.backward(retain_graph=True)
				torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
				self.state_optimizer.step()

		goals_predicted_debug_np = np.array(goals_predicted_debug)
		self.visualize_value_heatmaps_for_debug(goals_predicted_debug_np)

		return total_loss.item()




	
		
	def sample(self, episode_observes = None, qs = None):
		if episode_observes is None or qs is None:
			if self.latest_achieved is None:
				return np.array([0,8])
			goal_t =  self.generate_goal(self.latest_achieved,self.latest_rtgs + self.return_to_add)
			# trajectories, rtgs = self.best_trajectories.get_elements()
			# goal_t = self.generate_goal(episode_observes, rtgs[0] + self.return_to_add )
			goal =  goal_t.detach().cpu().numpy()
			self.latest_desired_goal = goal
			self.sampled_goals = np.concatenate((self.sampled_goals, [goal]))
			return goal
		else:
			episode_observes = np.array([self.eval_env.convert_obs_to_dict(episode_observes[i])["achieved_goal"] for i in range(len(episode_observes))])

			achieved_values, exploration_values, q_values, val_dict = self.get_rescaled_rewards(episode_observes, qs)

			rewards = self.gamma * achieved_values + self.beta * q_values + self.sigma * exploration_values# either get the qs earlier than 2000 steps or remove gamma limitation before then
			rtgs = self.reward_to_rtg(rewards)
   
			episode_observes = torch.tensor([episode_observes], device="cuda", dtype=torch.float32)
			rtgs = torch.tensor([rtgs], device="cuda", dtype=torch.float32)

			goal_t = self.generate_goal(episode_observes, rtgs + self.return_to_add)
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

	def get_probability_loss_over_trajectory(self, trajectory):
		probabilities = np.array([])
		for achieved_goal in trajectory:
			output = self.bnn_model(torch.tensor(achieved_goal, device = "cuda", dtype = torch.float32))
			probability = torch.sigmoid(output)
			uncertainty_loss = torch.abs(probability - 0.5)
			probabilities = np.append(probabilities,uncertainty_loss.unsqueeze(0).detach().cpu().numpy())
		return probabilities
	def get_probabilities_over_trajectory(self, trajectory):
		probabilities = np.array([])
		for achieved_goal in trajectory:
			output = self.bnn_model(torch.tensor(achieved_goal, device = "cuda", dtype = torch.float32))
			probability = torch.sigmoid(output)
			probabilities = np.append(probabilities,probability.unsqueeze(0).detach().cpu().numpy())
		return probabilities
	def create_combined_np(self):
		data_points = [
			[x, y]
			for x, y in itertools.product(
				range(self.limits[0][0], self.limits[0][1] + 1),
				range(self.limits[1][0], self.limits[1][1] + 1),
			)
		]
		return np.array(data_points, dtype=np.float32)
	def visualize_value_heatmaps_for_debug(self, goals_predicted_during_training):
		self.visualizer.visualize_value_heatmaps_for_debug(goals_predicted_during_training)

