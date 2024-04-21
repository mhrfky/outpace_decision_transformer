import copy
import numpy as np
from hgg.gcc_utils import gcc_load_lib, c_double, c_int
import torch
import torch.nn.functional as F
import time
from dt.models.decision_transformer import DecisionTransformer


def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a - goal_b, ord=2)
def goal_concat(obs, goal):
	return np.concatenate([obs, goal], axis=0)
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum
class TrajectoryPool:
	def __init__(self, pool_length):
		self.length = pool_length
		self.pool = []
		self.pool_init_state = []
		self.counter = 0

	def insert(self, trajectory, init_state):
		if self.counter<self.length:
			self.pool.append(trajectory.copy())
			self.pool_init_state.append(init_state.copy())
		else:
			self.pool[self.counter%self.length] = trajectory.copy()
			self.pool_init_state[self.counter%self.length] = init_state.copy()
		self.counter += 1

	def pad(self):
		if self.counter>=self.length:
			return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
		pool = copy.deepcopy(self.pool)
		pool_init_state = copy.deepcopy(self.pool_init_state)
		while len(pool)<self.length:
			pool += copy.deepcopy(self.pool)
			pool_init_state += copy.deepcopy(self.pool_init_state)
		return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

class DTSampler:    
	def __init__(self, goal_env, goal_eval_env, env_name, achieved_trajectory_pool, num_episodes,				
				agent = None, max_episode_timesteps =None,  normalize_aim_output=False,
				add_noise_to_goal= False, sigma = 1, gamma=0.99, device = 'cuda', critic = None, 
				dt : DecisionTransformer = None, rtg_optimizer = None, state_optimizer = None,
				loss_fn  = torch.nn.MSELoss(),
				):
		# Assume goal env
		self.env = goal_env
		self.eval_env = goal_eval_env
		self.env_name = env_name
		
		self.add_noise_to_goal = add_noise_to_goal
		self.agent = agent
		self.rtg_optimizer = rtg_optimizer
		self.state_optimizer = state_optimizer
		self.critic = critic

		self.max_episode_timesteps = max_episode_timesteps
		self.normalize_aim_output = normalize_aim_output
		self.gamma = gamma

		self.device = device
  		
		self.success_threshold = {'AntMazeSmall-v0' : 1.0, # 0.5,
								  'PointUMaze-v0' : 0.5,
		  						  'PointNMaze-v0' : 0.5,
								  'sawyer_peg_push' : getattr(self.env, 'TARGET_RADIUS', None),
								  'sawyer_peg_pick_and_place' : getattr(self.env, 'TARGET_RADIUS', None),
								  'PointSpiralMaze-v0' : 0.5,								  
								}
		self.loss_fn = torch.nn.MSELoss()

		self.dim = np.prod(self.env.convert_obs_to_dict(self.env.reset())['achieved_goal'].shape)
		self.delta = self.success_threshold[env_name] #self.env.distance_threshold
		self.goal_distance = goal_distance

		self.length = num_episodes 
		
		init_goal = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['achieved_goal'].copy()
		
		self.pool = np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
				
		self.achieved_trajectory_pool = achieved_trajectory_pool    

		self.latest_achieved = None

	def reward_to_rtg(self,rewards):
		rtg = discount_cumsum(rewards)

		return rtg #TODO check if it is working properly



	def sample(self, episode_observes):
		if episode_observes is None:
			pass # TODO work on latest_episode 
		else:
			pass # TODO work on the episode observes, instead of residual walking this might be better to implement

	def generate_achieved_value(self,achieved_pool_init_state,achieved_pool):
		# maybe for all timesteps in an episode
		obs = [goal_concat(achieved_pool_init_state, achieved_pool[j]) for  j in range(achieved_pool.shape[0])] # list of [dim] (len = ts)
																															# merge of achieved_pool and achieved_pool_init_state to draw trajectory
		
		with torch.no_grad(): ## when using no_grad, no gradients will be calculated or stored for operations on tensors, which can reduce memory usage and speed up computations				
			obs_t = torch.from_numpy(np.stack(obs, axis =0)).float().to(self.device) #[ts, dim]				
			if (self.agent.aim_discriminator is not None) and ('aim_f' in self.cost_type): # or value function is proxy for aim outputs
				value = -self.agent.aim_discriminator(obs_t).detach().cpu().numpy()[:, 0] # TODO discover inside aim_discriminator,
																							# 	* what kind of inputs it does require
																							# 	* value can be interpreted as a measure of how desirable or advantageous the current state is from the perspective of achieving the final goal
																						
				# value = np.clip(value, -1.0/(1.0-self.gamma), 0)
		return value
	
	def normalize_achieved_value(self,achieved_value):
		if 'aim_f' in self.cost_type: #normalizing achieved_values
			# print('normalize aim output in hgg update!')
			# For considering different traj length
			aim_outputs_max = -np.inf
			aim_outputs_min = np.inf
			for i in range(len(achieved_value)): # list of aim_output [ts,]
				if achieved_value[i].max() > aim_outputs_max:
					aim_outputs_max = achieved_value[i].max()
				if achieved_value[i].min() < aim_outputs_min:
					aim_outputs_min = achieved_value[i].min()
			for i in range(len(achieved_value)):
				achieved_value[i] = ((achieved_value[i]-aim_outputs_min)/(aim_outputs_max - aim_outputs_min+0.00001)-0.5)*2 #[0, 1] -> [-1,1]
				

	
	def update(self, achieved_goals, desired_goals, replay_buffer = None, meta_nml_epoch = 0):
		
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy(desired_goals)
			return
		
		#achieved pool has the whole trajectory throughtout the episode, while achieved_pool_init_state has the initial state where it started the episode
		achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad() # dont care about pad, it receives the stored achieved trajectories

		assert len(achieved_pool)>=self.length, 'If not, errors at assert match_count==self.length, e.g. len(achieved_pool)=5, self.length=25, match_count=5'
		if 'aim_f' in self.cost_type: 
			assert self.agent.aim_discriminator is not None


		achieved_value = self.generate_achieved_value(achieved_pool_init_state,achieved_pool)
		# TODO estimate Q-values as well 
  
		self.normalize_achieved_value(achieved_value)

		# TODO add training here
		
		self.latest_achieved = achieved_goals

	def train(self, achieved_goals, rtgs):
		pass
		#TODO just generate a simple training
		#TODO loop over all achieved_goals and predict on varying length of episodes

	def generate_goal(self,achieved_goals,rtgs, return_to_add):
		
		pass


