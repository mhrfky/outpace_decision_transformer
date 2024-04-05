import copy
import numpy as np
from hgg.gcc_utils import gcc_load_lib, c_double, c_int
import torch
import torch.nn.functional as F
import time
from dt.models.decision_transformer import DecisionTransformer
from outpacesac import OUTPACEAgent
def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a - goal_b, ord=2)
def goal_concat(obs, goal):
	return torch.concatenate([obs, goal], axis=0)



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
	def __init__(self, dt, achieved_trajectory_pool, num_episodes, add_noise_to_goal, normalize_aim_output, 
			  optimizer,device = 'cuda' , trajectory_to_be_shorten = False, agent = None, init_goal = None, 
			  loss_fn = torch.nn.MSELoss(), eval_env = None, original_final_goal = (0,0), env_lower_bound = (-2,-2),
			  env_upper_bound = (10,10)) -> None:
		self.dt : DecisionTransformer = dt
		self.achieved_trajectory_pool = achieved_trajectory_pool
		self.length = num_episodes
		self.add_noise_to_goal = add_noise_to_goal
		self.normalize_aim_output = normalize_aim_output
		self.device = device
		self.trajectory_to_be_shorten = trajectory_to_be_shorten 
		self.agent = agent
		self.loss_fn = loss_fn
		self.eval_env = eval_env
		self.optimizer = optimizer
		self.last_achieved = np.random.rand(100, 2) * (10 +2 ) -2
		self.gamma =  1
		self.alpha = 1
		self.lambda_ = 1
		self.sigma = 1
		self.agent = agent
		self.original_final_goal = original_final_goal
		self.env_lower_bound = env_lower_bound
		self.env_upper_bound = env_upper_bound
		# self.pool = np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))

	def convert_data_to_rtg(obs, rewards):
		pass
	def upscale_goal(self,goal):
		return np.array(self.scale_position(goal,((-1,-1),(1,1)),(self.env_lower_bound,self.env_upper_bound)))
	def downscale_goals(self,goals):
		for i in range(len(goals)):
			goals[i] = self.scale_position(goals[i],(self.env_lower_bound,self.env_upper_bound),((-1,-1),(1,1)))
		return goals
	def scale_value(self,value, old_min, old_max, new_min, new_max):
		"""Scales a value from one range to another."""
		return (value - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

	def scale_position(self, pos, old_bounds, new_bounds):
		"""Scales a 2D position from one range to another for both x and y components."""
		return (
			self.scale_value(pos[0], old_bounds[0][0], old_bounds[1][0], new_bounds[0][0], new_bounds[1][0]),
			self.scale_value(pos[1], old_bounds[0][1], old_bounds[1][1], new_bounds[0][1], new_bounds[1][1])
		)


	def custom_loss(self, achieved_goals_tensor, desired_goals_tensor):
		# Initialize the MSELoss function
		mse_loss_fn = torch.nn.MSELoss()
		
		# Compute achievability loss (Ensure both tensors are on the correct device and have the same dtype)
		achievability_loss = mse_loss_fn(achieved_goals_tensor, desired_goals_tensor)
		
		# Assuming generate_achieved_values returns a scalar or tensor that represents some form of loss or value
		# you want to minimize. Ensure it's computed correctly.
		achieved_value = self.generate_achieved_values(achieved_goals_tensor[0], achieved_goals_tensor[1:])
		achieved_value_loss = self.normalize_achieved_value(achieved_value)
		achieved_loss_tensor = torch.tensor(achieved_value_loss, dtype=torch.float32, requires_grad=True)
		
		# achieved_loss_tensor = -torch.mean(achieved_loss_tensor)
		achieved_loss_tensor = -torch.mean(achieved_loss_tensor)
		adjusted_achieved_aim_loss = achieved_loss_tensor * self.gamma
		adjusted_achievability_loss = achievability_loss * self.alpha
		# Combine loss components
		# Make sure both components are tensors and on the same device.
		# Adjust coefficients as necessary. Ensure alpha and gamma are defined and are tensors or scalars.
		total_loss = adjusted_achievability_loss+ adjusted_achieved_aim_loss
		print(adjusted_achievability_loss,adjusted_achieved_aim_loss,total_loss)

		return total_loss

	def compute_graph_pen(self):
		pass

	def train(self, achieved_goals, desired_goal,qs):
		

		desired_goals = [desired_goal for _ in achieved_goals]
		achieved_goals = self.downscale_goals(achieved_goals)
		desired_goals = self.downscale_goals(desired_goals)
		desired_goals = torch.tensor(desired_goals, dtype=torch.float32, requires_grad=True)
		achieved_goals = torch.tensor(achieved_goals, dtype=torch.float32, requires_grad=True)

		

		loss = self.custom_loss(achieved_goals,desired_goals)

		self.optimizer.zero_grad()
		loss.backward()

		# torch.nn.utils.clip_grad_norm_(self.dt.parameters(), .5)
		self.optimizer.step()		
	
	# def train_on_out_of_bounds(generated_goal):
	# 	tensor_goal = torch.tensor(generated_goal)


	def convert_observes_to_achieved_and_desired(self,episode_observes):
		list_of_achieved = [[obs[6],obs[7]] for obs in episode_observes]
		return list_of_achieved, [episode_observes[0][-2],episode_observes[0][-1]]
		
	def sample(self):
		return self.upscale_goal(self.generate_goal()) # TODO : decide whether to have a pool of goals by or one goal generated by dt
	def update(self,  episode_observes ,qs,replay_buffer = None):
		# if self.achieved_trajectory_pool.counter == 0:
		# 	self.pool = copy.deepcopy(desired_goals)
		# 	return
		# achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()

		# if self.trajectory_to_be_shorten:
			# achieved_pool = self.shorten_trajectory(achieved_pool = achieved_pool)
		
		# assert len(achieved_pool) >= self.length
		# assert self.agent.aim_discriminator is not None

		# achieved_values = self.generate_achieved_values(achieved_pool_init_state, achieved_pool)
		# self.normalize_achieved_value(achieved_values)
		list_of_achieved, desired_goal = self.convert_observes_to_achieved_and_desired(episode_observes)
		self.last_achieved = list_of_achieved
		self.train(list_of_achieved,desired_goal,qs)
		


	def generate_achieved_values(self,achieved_pool_init_state, achieved_pool):
		obs = [goal_concat(achieved_pool_init_state, achieved_pool[j]) for  j in range(achieved_pool.shape[0])] # list of [dim] (len = ts)																							# merge of achieved_pool and achieved_pool_init_state to draw trajectory
		
		with torch.no_grad(): ## when using no_grad, no gradients will be calculated or stored for operations on tensors, which can reduce memory usage and speed up computations				
			obs_t = torch.from_numpy(np.stack(obs, axis =0)).float().to(self.device) #[ts, dim]				
			if (self.agent.aim_discriminator is not None) :#and ('aim_f' in self.cost_type): # or value function is proxy for aim outputs
				value = -self.agent.aim_discriminator(obs_t).detach().cpu().numpy()[:, 0] 															
		
		return value
	
	def normalize_achieved_value(self,achieved_value):
		# print('normalize aim output in hgg update!')
		# For considering different traj length
		aim_outputs_max = -np.inf
		aim_outputs_min = np.inf
		if achieved_value.max() > aim_outputs_max:
			aim_outputs_max = achieved_value.max()
		if achieved_value.min() < aim_outputs_min:
			aim_outputs_min = achieved_value.min()
		achieved_value = ((achieved_value-aim_outputs_min)/(aim_outputs_max - aim_outputs_min+0.00001)-0.5)*2 #[0, 1] -> [-1,1]
		return achieved_value
	def shorten_trajectory():
		pass

	def generate_goal(self):
		if self.last_achieved is None:
			return self.original_final_goal
		if torch.cuda.is_available():

			timesteps = torch.arange(0,100).to(device="cuda")
			states = torch.tensor(self.last_achieved).to(device="cuda")
			rtg = torch.tensor([1000]).to(device="cuda")
		new_goal = self.dt.get_goal(states, rtg , timesteps)
		print(new_goal, end=" ")
		return new_goal.detach().cpu().numpy()
		
	
		
	