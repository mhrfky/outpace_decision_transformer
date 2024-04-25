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
	def __init__(self, goal_env, goal_eval_env, 				
				agent = None,
				add_noise_to_goal= False, sigma = 1, gamma=0.99, device = 'cuda', critic = None, 
				dt : DecisionTransformer = None, rtg_optimizer = None, state_optimizer = None,
				loss_fn  = torch.nn.MSELoss(),
				):
		# Assume goal env
		self.env = goal_env
		self.eval_env = goal_eval_env
		
		self.add_noise_to_goal = add_noise_to_goal
		self.agent = agent
		self.rtg_optimizer = rtg_optimizer
		self.state_optimizer = state_optimizer
		self.critic = critic

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
		self.goal_distance = goal_distance

		
		self.init_goal = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['achieved_goal'].copy()
		
				

		self.latest_achieved = None
		self.dt = dt
		self.loss_fn = loss_fn
		self.max_achieved_reward = 0
		self.return_to_add = 0.05
		self.discount_rate = 0.99 #TODO add this as init value
	def reward_to_rtg(self,rewards):
		rtg = discount_cumsum(rewards, self.discount_rate)

		return rtg #TODO check if it is working properly



	def sample(self, episode_observes = None):
		if episode_observes is None:
			if self.latest_achieved is None:
				return np.array([0.6,0.6])
			self.generate_goal(self.latest_achieved,[self.max_achieved_reward + self.return_to_add])
			pass # TODO work on latest_episode 
		else:
			pass # TODO work on the episode observes, instead of residual walking this might be better to implement

	def generate_achieved_value(self,init_state,achieved_goals):
		# maybe for all timesteps in an episode
		obs = [goal_concat(init_state, achieved_goals[j]) for  j in range(len(achieved_goals))] # list of [dim] (len = ts)
																															# merge of achieved_pool and achieved_pool_init_state to draw trajectory
		
		with torch.no_grad(): ## when using no_grad, no gradients will be calculated or stored for operations on tensors, which can reduce memory usage and speed up computations				
			obs_t = torch.from_numpy(np.stack(obs, axis =0)).float().to(self.device) #[ts, dim]				
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
				
	def get_q_value(self,obs):
		with torch.no_grad():
			dist = self.actor(obs)
			action = dist.rsample()

			q1, q2 = self.critic(obs, action)

			q_mean = torch.mean(q1,q2)
			# q_max = torch.max(q1,q2)
		return q_mean

	
	def update(self, achieved_goals, qs):
		
		
		#achieved pool has the whole trajectory throughtout the episode, while achieved_pool_init_state has the initial state where it started the episode
		# achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad() # dont care about pad, it receives the stored achieved trajectories
		if not len(qs):
			qs = np.zeros(101,)

		achieved_goals = np.array([self.eval_env.convert_obs_to_dict(achieved_goals[i])["achieved_goal"] for i in range(len(achieved_goals))])
		achieved_values= self.generate_achieved_value(self.init_goal,achieved_goals)
		rewards = self.gamma * achieved_values + (1-self.gamma) * qs
		rtgs = self.reward_to_rtg(rewards)
		qs = np.mean(qs, axis=0)
		# qs = np.min(qs, axis = 1)

		self.train(achieved_goals, rtgs)		
		self.latest_achieved = achieved_goals
		self.max_achieved_reward = max(rtgs)
		
	def train(self, achieved_goals, rtgs):
		if type(achieved_goals) != torch.tensor:
			achieved_goals = torch.tensor(achieved_goals, device="cuda", dtype=torch.float32)
		if type(rtgs) != torch.tensor:
			rtgs = torch.tensor(rtgs, device="cuda", dtype=torch.float32)
		actions = torch.zeros((100,2), device="cuda",dtype=torch.float32 )
		timesteps = torch.arange(0,100, device="cuda")
		for i in range(1,len(achieved_goals)):
			temp_actions = actions[:i].clone()
			temp_timesteps = timesteps[:i].clone()
			for j in range(i,len(achieved_goals)-1):
				temp_achieved = achieved_goals[:i].clone()
				temp_rtg = rtgs[:i].clone()
				temp_rtg -= rtgs[j:j+1]
				goal,_,_ = self.dt.forward(temp_achieved,temp_actions,None,temp_rtg,temp_timesteps)
				init_goal_pair = goal_concat(achieved_goals[0], goal)
				aim_val = self.agent.aim_discriminator(init_goal_pair)
				q_val = self.get_q_value(goal)
				goal_val = self.gamma * aim_val + q_val * (1-self.gamma)
				loss = torch.nn.L1Loss(goal_val,temp_rtg[0])

				self.state_optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
				self.optimizer.step()
			
		pass
		#TODO just generate a simple training
		#TODO loop over all achieved_goals and predict on varying length of episodes

	def generate_goal(self,achieved_goals,rtgs):
		actions = torch.zeros(len(achieved_goals))
		timesteps = torch.arange(0,100)
		desired_goal = self.dt.get_state(achieved_goals, actions, None, rtgs, timesteps)
		return desired_goal


