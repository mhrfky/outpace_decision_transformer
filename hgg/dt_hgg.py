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



def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a - goal_b, ord=2)
def goal_concat(obs, goal):
	return np.concatenate([obs, goal], axis=0)
def goal_concat_t(obs, goal):
	return torch.concatenate([obs, goal], axis=0)
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
		self.beta = -1
		self.sigma = 1

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
		
		self.init_goal = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['achieved_goal'].copy()
		
				
		self.final_goal = np.array([0,8]) #TODO make this parametrized
		self.latest_achieved = None
		self.dt = dt
		self.loss_fn = loss_fn
		self.max_achieved_reward = 0
		self.return_to_add = 0.05
		self.discount_rate = 1# 0.99 #TODO add this as init value
		self.limits = [[-2,8],[-2,8], [0,0]] # TODO make this generic
		self.latest_desired_goal = self.final_goal
		self.max_rewards_so_far = []
	def reward_to_rtg(self,rewards):
		return rewards - rewards[-1]




	def generate_achieved_values(self,init_state,achieved_goals):
		# maybe for all timesteps in an episode
		init_to_goal_pair	= 	[goal_concat(init_state, achieved_goals[j]) for  j in range(len(achieved_goals))] # list of [dim] (len = ts)
		goal_to_final_pair	= 	[goal_concat(achieved_goals[j], self.final_goal) for  j in range(len(achieved_goals))] # list of [dim] (len = ts) 																													# merge of achieved_pool and achieved_pool_init_state to draw trajectory
		
		with torch.no_grad(): ## when using no_grad, no gradients will be calculated or stored for operations on tensors, which can reduce memory usage and speed up computations				
			init_to_goal_pair_t  = torch.from_numpy(np.stack(init_to_goal_pair, axis =0)).float().to(self.device) #[ts, dim]				
			goal_to_final_pair_t = torch.from_numpy(np.stack(goal_to_final_pair, axis =0)).float().to(self.device) #[ts, dim]				
			values = -self.agent.aim_discriminator(init_to_goal_pair_t).detach().cpu().numpy()[:, 0] # TODO discover inside aim_discriminator,
																							# 	* what kind of inputs it does require
																							# 	* value can be interpreted as a measure of how desirable or advantageous the current state is from the perspective of achieving the final goal
			# values += self.agent.aim_discriminator(goal_to_final_pair_t).detach().cpu().numpy()[:, 0] 
				# value = np.clip(value, -1.0/(1.0-self.gamma), 0)
		return values
	
	def normalize_achieved_value(self,achieved_value):
		if 'aim_f' not in self.cost_type:
			return
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
				
	def get_q_values(self,goal_t):
		goal = goal_t.detach().cpu().numpy()
		with torch.no_grad():
			obs = self.eval_env.reset(goal=goal)
			obs_t = torch.tensor(obs, device = "cuda", dtype = torch.float32)
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
		
	def calculate_exploration_value(self, init_pos, curr_pos):
		epsilon = 1e-10  # Small value to prevent division by zero
		if type(init_pos) is torch.Tensor:
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
	def normalize_array(self,array):
		# Shift min to 0 by subtracting the minimum value
		shifted_array = array - np.min(array)

		# Scale to the range [0, 1]
		max_value 		= np.max(shifted_array)
		normalized_0_1	= shifted_array / max_value if max_value > 0 else shifted_array
		return 2 * normalized_0_1 - 1


	def shorten_trajectory(self, achieved_goals, rtgs):
		achieved_goals = achieved_goals[::3]# list of reduced ts		)
		rtgs = rtgs[::3]
  
		return achieved_goals, rtgs

	
	def update(self, step, episode, achieved_goals, qs):
		
		
		#achieved pool has the whole trajectory throughtout the episode, while achieved_pool_init_state has the initial state where it started the episode
		# achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad() # dont care about pad, it receives the stored achieved trajectories
		if not len(qs):
			qs = np.zeros((101,2))
		self.step = step
		self.episode = episode
		achieved_goals = np.array([self.eval_env.convert_obs_to_dict(achieved_goals[i])["achieved_goal"] for i in range(len(achieved_goals))])
		achieved_values= self.generate_achieved_values(achieved_goals[0],achieved_goals)
		exploration_vals = np.array([self.calculate_exploration_value(achieved_goals[0],achieved_goals[i]) for i in range(len(achieved_goals))])

		qs = self.normalize_array(qs)
		qs = np.min(qs, axis= 1)
		# achieved_values = self.normalize_array(achieved_values)
		exploration_vals = self.normalize_array(exploration_vals)
		
		rewards = self.gamma * achieved_values + self.beta * qs + self.sigma * exploration_vals# either get the qs earlier than 2000 steps or remove gamma limitation before then
		rtgs = self.reward_to_rtg(rewards)
		# qs = np.min(qs, axis = 1)

		achieved_goals, rtgs 	= self.shorten_trajectory(achieved_goals, rtgs)

		self.latest_achieved 	= achieved_goals
		self.latest_rtgs 		= rtgs
		
		self.train_single_trajectory(achieved_goals, rtgs)		
		self.max_achieved_reward = max(rtgs)
		self.max_rewards_so_far.append(self.max_achieved_reward)
		


	def train_single_trajectory(self, achieved_goals, rtgs):
		# Ensure input tensors are on the correct device and type
		achieved_goals = torch.tensor([achieved_goals], device="cuda", dtype=torch.float32)
		rtgs = torch.tensor([rtgs], device="cuda", dtype=torch.float32)

		# Placeholder for actions and sequence of timesteps
		actions = torch.zeros((1, achieved_goals.size(1), 2), device="cuda", dtype=torch.float32)
		timesteps = torch.arange(achieved_goals.size(1), device="cuda").unsqueeze(0)  # Adding batch dimension


		# Iterate over each state in the trajectory except the last one
		for i in range(1, achieved_goals.shape[1]):
			# Isolate the sub-sequence up to the current step
			temp_achieved = achieved_goals[:,:i]
			temp_actions = actions[:,:i]
			temp_timesteps = timesteps[:,:i]
			for j in range(i+1,achieved_goals.shape[1]):
				temp_rtg = rtgs[:,:i].clone()
				temp_rtg -= rtgs[0,j]
				temp_rtg = temp_rtg.unsqueeze(-1) 	
				# temp_rtg = rtgs[:,:i].clone()
				# temp_rtg = temp_rtg.unsqueeze(-1) 

				# Forward pass to predict the next goal
				# Assuming the last state's output (new goal) is what you want to compare against
				predicted_goal, _, _  	= self.dt.forward(temp_achieved, temp_actions, None, temp_rtg, temp_timesteps)#, attention_mask=attention_mask)
				predicted_goal 			= predicted_goal[0,-1]
				# Calculate the difference in RTG to simulate the value difference locations
				# Assuming each step predicts a goal for the next state
				if i < achieved_goals.shape[1] - 1:
					expected_val 			= temp_rtg[0,0,0]  # Calculate the expected RTG difference
     
					init_goal_pair 			= goal_concat_t(achieved_goals[0,0], predicted_goal)
					aim_val_t 				= self.agent.aim_discriminator(init_goal_pair)
     
					q1_t, q2_t				= self.get_q_values(predicted_goal)
					q_val_t					= torch.min(q1_t,q2_t)
     
					exploration_val 		= self.calculate_exploration_value(achieved_goals[0],predicted_goal)
     
					goal_val 				= self.gamma * aim_val_t + q_val_t * (self.beta) + self.sigma * exploration_val

					# Calculate loss between expected RTG difference and predicted RTG
					loss = torch.nn.L1Loss()(goal_val, expected_val.unsqueeze(0))
					
					# loss = torch.nn.MSELoss()(torch.tensor([0,8], device = "cuda", dtype= torch.float32), predicted_goal) # it can overfit just fine
					
					# Optimization step
					self.state_optimizer.zero_grad()
					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
					self.state_optimizer.step()
		self.visualize_value_heatmaps_for_debug()

		return loss.item()  # Return the last computed loss

	def sample(self, episode_observes = None):
		if episode_observes is None:
			if self.latest_achieved is None:
				return np.array([0,8])
			goal_t =  self.generate_goal(self.latest_achieved,[self.latest_rtgs + self.return_to_add])
			goal =  goal_t.detach().cpu().numpy()
			self.latest_desired_goal = goal

			return goal
		else:
			pass # TODO work on the episode observes, instead of residual walking this might be better to implement

	def generate_goal(self,achieved_goals,rtgs):
		actions = torch.zeros((1, achieved_goals.shape[0], 2), device="cuda", dtype=torch.float32)
		timesteps = torch.arange(achieved_goals.shape[0], device="cuda").unsqueeze(0)  # Adding batch dimension
		achieved_goals = torch.tensor([achieved_goals], device="cuda", dtype=torch.float32)

		rtgs = torch.tensor([rtgs], device = "cuda", dtype = torch.float32).unsqueeze(0)
		desired_goal = self.dt.get_state(achieved_goals, actions, None, rtgs, timesteps)
		return desired_goal



	def visualize_value_heatmaps_for_debug(self):
		assert self.video_recorder is not None
		fig_shape = (3,3)
		fig, axs = plt.subplots(fig_shape[0],fig_shape[1], figsize=(16, 6))  # 1 row, 2 columns of subplots
		combined_heatmap = self.create_combined_np() # type: ignore

		plot_dict = {}
		plot_dict["Q Heatmap"], q1_vals, q2_vals  = self.generate_q_values_for_heatmap()
		plot_dict["Aim Heatmap"]  = self.generate_aim_values_for_heatmap()
		plot_dict["Explore Heatmap"]  = self.generate_exploration_values_for_heatmap()

		for heatmap in plot_dict.values():
			combined_heatmap[:,2] += heatmap[:,2]

		plot_dict["Combined Heatmap"] = combined_heatmap
		plot_dict["Q1 Heatmap"], plot_dict["Q2 Heatmap"] = q1_vals, q2_vals

		for i,key in enumerate(plot_dict.keys()):

			pos = (i // fig_shape[1],i % fig_shape[1])
			self.plot_heatmap(plot_dict[key],axs[pos[0]][pos[1]],key)

		i +=1
		pos = (i // fig_shape[1],i % fig_shape[1])
		self.visualize_trajectories_on_time(axs[pos[0]][pos[1]])

		i +=1
		pos = (i // fig_shape[1],i % fig_shape[1])
		self.visualize_trajectories_on_rtgs(axs[pos[0]][pos[1]])
  
		i +=1
		pos = (i // fig_shape[1],i % fig_shape[1])
		self.visualize_max_rewards(axs[pos[0]][pos[1]])

		plt.savefig(
			f'{self.video_recorder.debug_dir}/combined_heatmaps_episode_{str(self.episode)}.jpg'
		)
		plt.close(fig)
	def visualize_max_rewards(self, ax):
		max_rewards_np = np.array(self.max_rewards_so_far)
		x = np.arange(0, len(max_rewards_np))
		ax.plot(x, max_rewards_np, label='Trend', marker='o', markersize=5, linestyle='-', linewidth=2)
		ax.set_xlabel('x')
		ax.set_ylabel('reward')
		ax.set_title('Max Rewards')
		# ax.set_aspect('equal')  # Ensuring equal aspect ratio

		ax.grid(True)  # Enable grid for better readability


	def create_combined_np(self):
		data_points = [
			[x, y, 0.0]
			for x, y in itertools.product(
				range(self.limits[0][0], self.limits[0][1]),
				range(self.limits[1][0], self.limits[1][1]),
			)
		]
		return np.array(data_points, dtype=np.float32)
	
	def generate_aim_values_for_heatmap(self):
		data_points = []
		i = 0
		for x in range(self.limits[0][0], self.limits[0][1]):
			for y in range(self.limits[1][0], self.limits[1][1]):
				pos = np.array([x, y])
				init_to_goal_pair	= [goal_concat(self.init_goal, pos)]
				goal_to_final_pair	= [goal_concat(pos, self.final_goal)]
				with torch.no_grad():
					init_to_goal_pair_t = torch.from_numpy(np.stack(init_to_goal_pair, axis=0)).float().to(self.device)
					goal_to_final_pair_t = torch.from_numpy(np.stack(init_to_goal_pair, axis=0)).float().to(self.device)

					aim_val = -self.agent.aim_discriminator(init_to_goal_pair_t).detach().cpu().numpy()[:, 0]
					# aim_val += self.agent.aim_discriminator(goal_to_final_pair_t).detach().cpu().numpy()[:, 0] 
				data_points.append([x, y, self.gamma * aim_val])
				i+= 1
		data_points = np.array(data_points,  dtype=np.float32)
		data_points[:,2] = self.normalize_array(data_points[:,2])
		return data_points

	def generate_q_values_for_heatmap(self):
		q1_data_points = []
		q2_data_points = []
		q_min_data_points = []
		i = 0

		for x in range(self.limits[0][0], self.limits[0][1]):
			for y in range(self.limits[1][0], self.limits[1][1]):
				pos = torch.tensor([x, y], device=self.device)
				q1_t,q2_t = self.get_q_values(pos)
				q_val_t = torch.min(q1_t,q2_t)
				q_val = q_val_t.detach().cpu().numpy()
				q1 = q1_t.detach().cpu().numpy()
				q2 = q2_t.detach().cpu().numpy()
				q_min_data_points.append([x, y, self.beta *q_val])
				q1_data_points.append([x, y, self.beta *q1])
				q2_data_points.append([x, y, self.beta *q2])
				i+= 1
		q_min_data_points = np.array(q_min_data_points,  dtype=np.float32)
		q_min_data_points[:,2] = self.normalize_array(q_min_data_points[:,2])
		q1_data_points = np.array(q1_data_points,  dtype=np.float32)
		q1_data_points[:,2] = self.normalize_array(q1_data_points[:,2])
		q2_data_points = np.array(q2_data_points,  dtype=np.float32)
		q2_data_points[:,2] = self.normalize_array(q2_data_points[:,2])
		return q_min_data_points, q1_data_points, q2_data_points

	
	def generate_exploration_values_for_heatmap(self):
		data_points = []
		i = 0

		for x in range(self.limits[0][0], self.limits[0][1]):
			for y in range(self.limits[1][0], self.limits[1][1]):
				pos = [x,y]
				exploration_val = self.calculate_exploration_value(self.init_goal, pos)
				data_points.append([x, y, exploration_val])
				i+= 1
		return np.array(data_points,  dtype=np.float32)


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
		ax.set_xlim(-2, 8)
		ax.set_ylim(-2, 8)
		ax.set_aspect('equal')  # Ensuring equal aspect ratio
		ax.grid(True)

	def visualize_trajectories_on_time(self, ax, title='Position Over Time'):
		x = self.latest_achieved[:, 0]
		y = self.latest_achieved[:, 1]
		t = np.arange(0, len(x))

		t_normalized = (t - t.min()) / (t.max() - t.min())
		scatter = ax.scatter(x, y, c=t_normalized, cmap='viridis', edgecolor='k')
		ax.scatter(self.latest_desired_goal[0], self.latest_desired_goal[1], color='red', marker='x', s=100, label='Latest Desired Goal')

		cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')

		ax.set_xlabel('X Position')
		ax.set_ylabel('Y Position')
		ax.set_title(title)
		ax.set_xlim(-5, 10)
		ax.set_ylim(-5, 10)
		ax.set_aspect('equal')  # Ensuring equal aspect ratio
		ax.grid(True)
	def visualize_trajectories_on_rtgs(self, ax, title='Position Over RTGs'):
		x = self.latest_achieved[:, 0]
		y = self.latest_achieved[:, 1]
		rtgs = self.latest_rtgs

		# Normalize time values to [0, 1] for color mapping
		# t_normalized = (t - t.min()) / (t.max() - t.min())
		scatter = ax.scatter(x, y, c=rtgs, cmap='viridis', edgecolor='k')
		ax.scatter(self.latest_desired_goal[0], self.latest_desired_goal[1], color='red', marker='x', s=100, label='Latest Desired Goal')

		cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')

		ax.set_xlabel('X Position')
		ax.set_ylabel('Y Position')
		ax.set_title(title)
		ax.set_xlim(-5, 10)
		ax.set_ylim(-5, 10)
		ax.set_aspect('equal')  # Ensuring equal aspect ratio
		ax.grid(True)