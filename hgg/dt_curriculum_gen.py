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
       
def reward_to_rtg(rewards):
    rewards -= rewards[0]
    return rewards[::-1].copy()
class DTCurriculumGenerator:    
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
        self.curriculum = np.array([[0,0]])
		

    def update(self, step, episode, achieved_states, actions, qs):
        
        self.step           = step
        self.episode        = episode   
        achieved_states     = np.array([self.eval_env.convert_obs_to_dict(achieved_states[i])["achieved_goal"] for i in range(len(achieved_states))])
        actions             = np.array([self.eval_env.convert_obs_to_dict(achieved_states[i])["desired_goal" ] for i in range(len(achieved_states))])

        rewards             = self.eval_env.get_rewards(achieved_states, qs)
        rtgs                = reward_to_rtg(rewards)

        self.latest_achieved 	= np.array([achieved_states])
        self.latest_acts        = np.array([actions])
        self.latest_rtgs 		= np.array([rtgs ])

        self.max_rewards_so_far.append(max(rewards))
        self.max_achieved_reward = max(max(rewards),self.max_achieved_reward)

        if self.episode % self.log_every_n_times == 0:
            proclaimed_states, _ , proclaimed_rtgs, _ = self.generate_next_n_states(self.latest_achieved[:,:20], self.latest_acts[:,:20], np.expand_dims(self.latest_rtgs[:,:20],axis=-1), torch.arange(20, device="cuda").unsqueeze(0), n = 90)
            proclaimed_states = proclaimed_states.squeeze(0).detach().cpu().numpy() 
            proclaimed_rtgs = proclaimed_rtgs.squeeze(0).detach().cpu().numpy()
            # self.visualize_value_heatmaps_for_debug(goals_predicted_debug_np, proclaimed_states, proclaimed_rtgs)
            self.residual_this_episode = False
            self.debug_trajectories = []

    def generate_till_the_desired(self,states_t, actions_t, rtgs_t, t, n = 10):
        states_t = states_t.squeeze(0)
        actions_t = actions_t.squeeze(0)
        rtgs_t = rtgs_t.squeeze(0)
        t_t = t.squeeze(0)
        while torch.dist(states_t[0,-1], actions_t[0,-1]) > 0.5 and t_t[-1] < 100:
            states_t, actions_t, rtgs_t = self.generate_next_n_states(states_t, actions_t, rtgs_t, t, n = 1)
            t = torch.tensor([t[-1] + 1], device = 'cuda').unsqueeze(0)
        return states_t, actions_t, rtgs_t
    
    def train(self):
        for i in range(self.num_of_train_per_iteration):
            self.train_step()
    
    def train_last(self):
        achieved_states = self.latest_achieved
        actions = self.latest_acts
        rtgs = self.latest_rtgs

                    