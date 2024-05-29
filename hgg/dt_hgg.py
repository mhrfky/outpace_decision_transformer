from queue import PriorityQueue
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
class TrajectoryHeap:
	class Trajectory:
		def __init__(self, trajectory, rtgs) -> None:
			self.trajectory = trajectory
			self.rtgs = rtgs
		def __lt__(self, other):
			return self.rtgs[0] < other.rtgs[0]

	def __init__(self, max_size = 5):
		self.max_size = max_size
		self.heap = []

	def add(self, achieved_goals, rtgs):
		traj = self.Trajectory(achieved_goals,rtgs)
		if len(self.heap) < self.max_size:
			heapq.heappush(self.heap, traj)
		elif traj > self.heap[0]:
			heapq.heapreplace(self.heap, traj)
		return self.get_elements()

	def get_elements(self):
		trajectories = [traj.trajectory for traj in self.heap]
		rtgs_s = [traj.rtgs for traj in self.heap]
		return trajectories, rtgs_s

def rescale_array(tensor, old_min, old_max, new_min =-1, new_max = 1):
    # rescaled_tensor = (tensor - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return tensor

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
		self.residual_goals_debug = []
		self.best_trajectories =  TrajectoryHeap(max_size=5)
		self.number_of_trajectories_per_episode = 2
		self.dumb_prevention_loss = MoveOntheLastPartLoss(threshold=4)
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

  
	def update(self, step, episode, achieved_goals, qs):
		
		
		#achieved pool has the whole trajectory throughtout the episode, while achieved_pool_init_state has the initial state where it started the episode
		# achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad() # dont care about pad, it receives the stored achieved trajectories
		if not len(qs):
			qs = np.zeros((101,2))
		self.step = step
		self.episode = episode
		achieved_goals = np.array([self.eval_env.convert_obs_to_dict(achieved_goals[i])["achieved_goal"] for i in range(len(achieved_goals))])


		achieved_values, exploration_values, q_values, min_max_val_dict = self.get_rescaled_rewards(achieved_goals, qs)
		rewards = self.gamma * achieved_values + self.beta * q_values + self.sigma * exploration_values# either get the qs earlier than 2000 steps or remove gamma limitation before then

		rtgs = self.reward_to_rtg(rewards)


		self.latest_achieved 	= np.array([achieved_goals])
		self.latest_rtgs 		= np.array([rtgs])
  
		# achieved_goalss, rtgss = self.best_trajectories.add(achieved_goals,rtgs)
		# rand_ind = random.randint(0, len(self.best_trajectories.heap)-1)
		# for i in range(len(achieved_goalss)):
		self.train_single_trajectory(achieved_goals, rtgs, min_max_val_dict)
   		
		self.max_achieved_reward = max(max(rtgs),self.max_achieved_reward)
		self.max_rewards_so_far.append(self.max_achieved_reward)
		self.residual_goals_debug = []
		
		

	def train_single_trajectory(self, achieved_goals, rtgs, min_max_val_dict):
		
		goals_predicted_debug = []
		
		# Ensure input tensors are on the correct device and type
		achieved_goals = torch.tensor([achieved_goals], device="cuda", dtype=torch.float32)
		rtgs = torch.tensor([rtgs], device="cuda", dtype=torch.float32)

		# Placeholder for actions and sequence of timesteps
		actions = torch.zeros((1, achieved_goals.size(1), 2), device="cuda", dtype=torch.float32)
		timesteps = torch.arange(achieved_goals.size(1), device="cuda").unsqueeze(0)  # Adding batch dimension


		# Iterate over each state in the trajectory except the last one
		for i in range(1, achieved_goals.shape[1]-1):
			# Isolate the sub-sequence up to the current step
			temp_achieved = achieved_goals[:,:i]
			temp_actions = actions[:,:i]
			temp_timesteps = timesteps[:,:i]
			
			temp_rtg = rtgs[:,:i].clone()
			temp_rtg -= rtgs[0,i+1]
			temp_rtg = temp_rtg.unsqueeze(-1) 	


			# Forward pass to predict the next goal
			# Assuming the last state's output (new goal) is what you want to compare against
			predicted_goal, _, predicted_return  	= self.dt.forward(temp_achieved, temp_actions, None, temp_rtg, temp_timesteps)#, attention_mask=attention_mask)
			predicted_goal 			= predicted_goal[0,-1]
			predicted_return 		= predicted_return[0,-1]
			goals_predicted_debug.append(predicted_goal.detach().cpu().numpy())
			# Calculate the difference in RTG to simulate the value difference locations
			# Assuming each step predicts a goal for the next state
			if i < achieved_goals.shape[1] - 1:
				expected_val 			= temp_rtg[0,0,0]  # Calculate the expected RTG difference

				init_goal_pair 			= goal_concat_t(achieved_goals[0,0], predicted_goal)
				aim_val_t 				= self.agent.aim_discriminator(init_goal_pair)

				q1_t, q2_t				= self.get_q_values(predicted_goal)
				q_val_t					= torch.min(q1_t,q2_t)

				exploration_val 		= self.calculate_exploration_value(achieved_goals[0],predicted_goal)
				aim_val_t 				= rescale_array(aim_val_t, min_max_val_dict["min_aim"],  min_max_val_dict["max_aim"])
				exploration_val 		= rescale_array(exploration_val, min_max_val_dict["min_expl"],  min_max_val_dict["max_expl"])
				q_val_t			 		= rescale_array(q_val_t, min_max_val_dict["min_q"],  min_max_val_dict["max_q"])
				goal_val_t 				= self.gamma * aim_val_t + q_val_t * self.beta + self.sigma * exploration_val
				best_state_index = torch.argmin(temp_rtg[0])
				
				# Calculate loss between expected RTG difference and predicted RTG
				rtg_pred_loss  					= torch.nn.L1Loss()(goal_val_t, expected_val.unsqueeze(0))
				state_pred_loss					= self.chi_distance_loss(predicted_goal, achieved_goals[:,i], 1.0)
				# rtg_gain_reward					= torch.nn.L1Loss()(goal_val ,(rtgs[0,0] - rtgs[0,best_state_index]).unsqueeze(-1))
				dumb_loss = self.dumb_prevention_loss.forward(temp_achieved,predicted_goal)
				total_loss = rtg_pred_loss + state_pred_loss - q_val_t#- goal_val_t #+ dumb_loss
				# loss = torch.nn.MSELoss()(torch.tensor([0,8], device = "cuda", dtype= torch.float32), predicted_goal) # it can overfit just fine
				# print(f"rtg gain reward : {rtg_gain_reward},\nstate_pred_loss : {state_pred_loss}\nrtg_pred_loss : {rtg_pred_loss}, {goal_val} + {predicted_return} = {expected_val}")
				# Optimization step
				self.state_optimizer.zero_grad()
				total_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
				self.state_optimizer.step()
		goals_predicted_debug_np = np.array(goals_predicted_debug)
		self.visualize_value_heatmaps_for_debug(goals_predicted_debug_np)

		return total_loss.item()  # Return the last computed loss
	def chi_distance_loss(self, a, b, demanded_dist):
		dist = torch.sqrt(torch.sum((a - b) ** 2))
		loss = (demanded_dist - dist) ** 2
		return loss

	
		
	def sample(self, episode_observes = None, qs = None):
		if episode_observes is None or qs is None:
			if self.latest_achieved is None:
				return np.array([0,8])
			goal_t =  self.generate_goal(self.latest_achieved,self.latest_rtgs + self.return_to_add)
			# trajectories, rtgs = self.best_trajectories.get_elements()
			# goal_t = self.generate_goal(episode_observes, rtgs[0] + self.return_to_add )
			goal =  goal_t.detach().cpu().numpy()
			self.latest_desired_goal = goal

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


	def generate_goal(self,achieved_goals,rtgs):
		actions = torch.zeros((1, achieved_goals.shape[0], 2), device="cuda", dtype=torch.float32)
		timesteps = torch.arange(achieved_goals.shape[0], device="cuda").unsqueeze(0)  # Adding batch dimension
		achieved_goals = torch.tensor(achieved_goals, device="cuda", dtype=torch.float32)

		rtgs = torch.tensor(rtgs, device = "cuda", dtype = torch.float32).unsqueeze(0)
		desired_goal = self.dt.get_state(achieved_goals, actions, None, rtgs, timesteps)
  
		return desired_goal



	def visualize_value_heatmaps_for_debug(self, goals_predicted_during_training):
		assert self.video_recorder is not None
		fig_shape = (3,3)
		fig, axs = plt.subplots(fig_shape[0], fig_shape[1], figsize=(16, 12), constrained_layout=True)  # 3 rows, 3 columns of subplots
		combined_heatmap = self.create_combined_np()  # type: ignore
		achieved_values, exploration_values, q_values, _ = self.get_rescaled_rewards(combined_heatmap, None)
		achieved_values = achieved_values.reshape(-1, 1)
		exploration_values = exploration_values.reshape(-1, 1)
		q_values = q_values.reshape(-1, 1)
		q_pos_val = np.hstack((combined_heatmap, self.beta * q_values))
		aim_pos_val = np.hstack((combined_heatmap, self.gamma * achieved_values))
		expl_pos_val = np.hstack((combined_heatmap, self.sigma * exploration_values))
		combined_pos_val = np.hstack((combined_heatmap, (self.beta * q_values +  self.gamma * achieved_values + self.sigma * exploration_values)))
		plot_dict = {}
		plot_dict["Q Heatmap"] = q_pos_val
		plot_dict["Aim Heatmap"]  = aim_pos_val
		plot_dict["Explore Heatmap"]  = expl_pos_val
		plot_dict["Combined Heatmap"] = combined_pos_val
		
  		# for heatmap in plot_dict.values():
		# 	combined_heatmap[:, 2] += heatmap[:, 2]

		# plot_dict["Combined Heatmap"] = combined_heatmap

		for i, key in enumerate(plot_dict.keys()):
			pos = (i // fig_shape[1], i % fig_shape[1])
			self.plot_heatmap(plot_dict[key], axs[pos[0]][pos[1]], key)
			axs[pos[0]][pos[1]].scatter(goals_predicted_during_training[:,0],goals_predicted_during_training[:,1], c = np.arange(len(goals_predicted_during_training)), cmap = "gist_heat", s=10)

		i += 1
		pos = (i // fig_shape[1], i % fig_shape[1])
		self.visualize_trajectories_on_time(axs[pos[0]][pos[1]])

		i += 1
		pos = (i // fig_shape[1], i % fig_shape[1])
		self.visualize_trajectories_on_rtgs(axs[pos[0]][pos[1]])

		i += 1
		pos = (i // fig_shape[1], i % fig_shape[1])
		self.visualize_max_rewards(axs[pos[0]][pos[1]])

		i += 1
		pos = (i // fig_shape[1], i % fig_shape[1])
		self.visualize_best_trajectories(axs[pos[0]][pos[1]], "Trajectories 1")

		# i += 1
		# pos = (i // fig_shape[1], i % fig_shape[1])
		# self.visualize_best_trajectories(axs[pos[0]][pos[1]], "Trajectories 2")

		plt.savefig(
			f'{self.video_recorder.debug_dir}/combined_heatmaps_episode_{str(self.episode)}.jpg'
		)
		plt.close(fig)

	def visualize_max_rewards(self, ax):
		max_rewards_np = np.array(self.max_rewards_so_far)
		x = np.arange(0, len(max_rewards_np))
		ax.plot(x, max_rewards_np, label='Trend', marker='.', markersize=5, linestyle='-', linewidth=2)
		ax.set_xlabel('x')
		ax.set_ylabel('reward')
		ax.set_title('Max Rewards')
		ax.grid(True)  # Enable grid for better readability

	def visualize_best_trajectories(self, ax, title):
		trajectories, rtgs = self.best_trajectories.get_elements()
		colors = ['b', 'g', 'r', 'yellow', 'purple']
		for i, traj in enumerate(trajectories):
			x = traj[:, 0]
			y = traj[:, 1]
			ax.plot(x, y, marker='o', linestyle='-', color=colors[i])
		ax.set_title(title)
		ax.set_xlabel('X Position')
		ax.set_ylabel('Y Position')
		ax.set_xlim(-2, 8)
		ax.set_ylim(-2, 8)    
		ax.grid(True)

	def create_combined_np(self):
		data_points = [
			[x, y]
			for x, y in itertools.product(
				range(self.limits[0][0], self.limits[0][1] + 1),
				range(self.limits[1][0], self.limits[1][1] + 1),
			)
		]
		return np.array(data_points, dtype=np.float32)

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
		x = self.latest_achieved[0,:, 0]
		y = self.latest_achieved[0,:, 1]
		t = np.arange(0, len(x))


		t_normalized = (t - t.min()) / (t.max() - t.min())
		scatter = ax.scatter(x, y, c=t_normalized, cmap='viridis', edgecolor='k')
		if len(self.residual_goals_debug):
			residual_goals_np = np.array(self.residual_goals_debug)
			res_x = residual_goals_np[:,0]
			res_y = residual_goals_np[:,1]
			ax.scatter(res_x, res_y, color='blue', marker='x', s=100, label='Latest Desired Goal')
		ax.scatter(self.latest_desired_goal[0], self.latest_desired_goal[1], color='red', marker='x', s=100, label='Latest Desired Goal')

		cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')

		ax.set_xlabel('X Position')
		ax.set_ylabel('Y Position')
		ax.set_title(title)
		ax.set_xlim(-2, 8)
		ax.set_ylim(-2, 8)
		ax.set_aspect('equal')  # Ensuring equal aspect ratio
		ax.grid(True)
	def visualize_trajectories_on_rtgs(self, ax, title='Position Over RTGs'):
		x = self.latest_achieved[0,:, 0]
		y = self.latest_achieved[0,:, 1]
		rtgs = self.latest_rtgs[0]
		if self.residual_goals_debug:
			residual_goals_np = np.array(self.residual_goals_debug)
			res_x = residual_goals_np[:,0]
			res_y = residual_goals_np[:,1]
			ax.scatter(res_x, res_y, color='blue', marker='x', s=100, label='Latest Desired Goal')
  
		# Normalize time values to [0, 1] for color mapping
		# t_normalized = (t - t.min()) / (t.max() - t.min())
		scatter = ax.scatter(x, y, c=rtgs, cmap='viridis', edgecolor='k')
		ax.scatter(self.latest_desired_goal[0], self.latest_desired_goal[1], color='red', marker='x', s=100, label='Latest Desired Goal')
		ax.scatter(self.latest_desired_goal[0], self.latest_desired_goal[1], color='red', marker='x', s=100, label='Latest Desired Goal')
  
		if len(self.residual_goals_debug):
			residual_goals_np = np.array(self.residual_goals_debug)
			res_x = residual_goals_np[:,0]
			res_y = residual_goals_np[:,1]
			ax.scatter(res_x, res_y, color='blue', marker='x', s=100, label='Latest Desired Goal')
		cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')

		ax.set_xlabel('X Position')
		ax.set_ylabel('Y Position')
		ax.set_title(title)
		ax.set_xlim(-2, 8)
		ax.set_ylim(-2, 8)
		ax.set_aspect('equal')  # Ensuring equal aspect ratio
		ax.grid(True)