import os
from random import random, uniform
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'

from hgg.dt_hgg import DTSampler
from dt.models.decision_transformer import DecisionTransformer
import copy
import pickle as pkl
import sys
import time
import numpy as np
from queue import Queue
import hydra
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer, HindsightExperienceReplayWrapperVer2
from video import VideoRecorder
import matplotlib.pyplot as plt
import seaborn as sns
from hgg.hgg import goal_distance
from visualize.visualize_2d import *
from debug_utils import time_decorator
torch.backends.cudnn.benchmark = True

class UniformFeasibleGoalSampler:
    def __init__(self, env_name):        
        self.env_name = env_name        
        if env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
            self.LOWER_CONTEXT_BOUNDS = np.array([-2, -2]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([10, 10])
        elif env_name in ['sawyer_peg_pick_and_place']:
            self.LOWER_CONTEXT_BOUNDS = np.array([-0.6, 0.2, 0.01478]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([0.6, 1.0, 0.4])            
        elif env_name in ['sawyer_peg_push']:
            self.LOWER_CONTEXT_BOUNDS = np.array([-0.6, 0.2, 0.01478]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([0.6, 1.0, 0.02])        
        elif env_name == "PointSpiralMaze-v0":
            self.LOWER_CONTEXT_BOUNDS = np.array([-10, -10]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([10, 10])
        elif env_name in ["PointNMaze-v0"]:
            self.LOWER_CONTEXT_BOUNDS = np.array([-2, -2]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([10, 18])
        else:
            raise NotImplementedError

    def is_feasible(self, context): 
        # Check that the context is not in or beyond the outer wall
        if self.env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']: # 0.5 margin
            if np.any(context < -1.5) or np.any(context > 9.5):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 6.5), np.logical_and(1.5 < context[1], context[1] < 6.5))):
                return False
            else:
                return True
        elif self.env_name == "PointSpiralMaze-v0":
            if np.any(context < -9.5) or np.any(context > 9.5):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 6.5), np.logical_and(1.5 < context[1], context[1] < 6.5))):
                return False
            elif np.all((np.logical_and(-6.5 < context[0], context[0] < -1.5), np.logical_and(-6.5 < context[1], context[1] < 6.5))):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 10.5), np.logical_and(-6.5 < context[1], context[1] < -1.5))):
                return False
            else:
                return True
            
        elif self.env_name in ["PointNMaze-v0"]:
            if (context[0] < -1.5) or (context[0] > 9.5):
                return False
            elif (context[1] < -1.5) or (context[1] > 17.5):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 6.5), np.logical_and(1.5 < context[1], context[1] < 6.5))):
                return False
            elif np.all((np.logical_and(1.5 < context[0], context[0] < 10.5), np.logical_and(9.5 < context[1], context[1] < 14.5))):
                return False
            else:
                return True
        
        elif self.env_name in ['sawyer_peg_pick_and_place']:
            if not np.all(np.logical_and(self.LOWER_CONTEXT_BOUNDS < context, context <self.UPPER_CONTEXT_BOUNDS)):
                return False            
            else:
                return True
        elif self.env_name in ['sawyer_peg_push']:
            if not np.all(np.logical_and(self.LOWER_CONTEXT_BOUNDS < context, context <self.UPPER_CONTEXT_BOUNDS)):
                return False            
            else:
                return True
        else:
            raise NotImplementedError

    def sample(self):
        
        sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        while not self.is_feasible(sample):
            sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        
        return sample


def get_object_states_only_from_goal(env_name, goal):
    if env_name in ['sawyer_door', 'sawyer_peg']:
        return goal[..., 4:7]

    elif env_name == 'tabletop_manipulation':
        raise NotImplementedError
    
    else:
        raise NotImplementedError

def get_original_final_goal(env_name):
    if env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
        original_final_goal = np.array([0., 8.])
    elif env_name in ['sawyer_peg_push']:
        original_final_goal = np.array([-0.3, 0.4, 0.02])
    elif env_name in ['sawyer_peg_pick_and_place']:
        original_final_goal = np.array([-0.3, 0.4, 0.2])
    elif env_name == "PointSpiralMaze-v0":
        original_final_goal = np.array([8., -8.])
    elif env_name in ["PointNMaze-v0"]:
        original_final_goal = np.array([8., 16.])
    else:
        raise NotImplementedError
    return original_final_goal.copy()


max_episode_timesteps_dict = {'AntMazeSmall-v0' : 300,
                              'PointUMaze-v0' : 100,
                              'sawyer_peg_pick_and_place' : 200,
                              'sawyer_peg_push' : 200,
                              'PointNMaze-v0' : 100, 
                              'PointSpiralMaze-v0' : 200,
                             }

num_seed_steps_dict = { 'AntMazeSmall-v0' : 4000,
                        'PointUMaze-v0' : 2000,
                        'sawyer_peg_pick_and_place' : 2000,
                        'sawyer_peg_push' : 2000,
                        'PointNMaze-v0' : 2000, 
                        'PointSpiralMaze-v0' : 2000,
                        }

num_random_steps_dict = {'AntMazeSmall-v0' : 4000,
                         'PointUMaze-v0' : 2000,
                         'sawyer_peg_pick_and_place' : 2000,
                         'sawyer_peg_push' : 2000,
                         'PointNMaze-v0' : 2000, 
                         'PointSpiralMaze-v0' : 2000,
                        }

randomwalk_random_noise_dict = {'AntMazeSmall-v0' : 2.5,
                                'PointUMaze-v0' : 2.5,
                                'sawyer_peg_pick_and_place' : 0.1,
                                'sawyer_peg_push' : 0.1,
                                'PointNMaze-v0' : 2.5, 
                                'PointSpiralMaze-v0' : 2.5,
                                }
aim_disc_replay_buffer_capacity_dict = {'AntMazeSmall-v0' : 50000,
                                        'PointUMaze-v0' : 10000,
                                        'sawyer_peg_pick_and_place' : 30000,
                                        'sawyer_peg_push' : 30000,
                                        'PointNMaze-v0' : 10000, 
                                        'PointSpiralMaze-v0' : 20000,
                                        }



class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.model_dir = utils.make_dir(self.work_dir, 'model')
        
        self.buffer_dir = utils.make_dir(self.work_dir, 'buffer')

        self.cfg = cfg
        self.episode = 0
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             action_repeat=cfg.action_repeat,
                             agent='outpace_rl')

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        obs_spec, action_spec = self.init_env(cfg)
            

        
        self.set_cfg_agent(cfg, obs_spec, action_spec)
        
            
        self.max_episode_timesteps = cfg.max_episode_timesteps
        
        if cfg.aim_discriminator_cfg.output_activation in [None, 'none', 'None']:
            cfg.aim_discriminator_cfg.output_activation = None
      
            

        self.meta_nml_init(cfg)
        
        self.init_feature_dim_for_discriminator_actor_critic(cfg)
        
            

        

        # exploration agent uses intrinsic reward
        self.expl_agent = hydra.utils.instantiate(cfg.agent)
        
            
        self.init_buffers(cfg, obs_spec, action_spec)
        if cfg.use_hgg:
            from hgg.hgg import TrajectoryPool, MatchSampler            
            self.init_hgg_achieved_trajectory_pool(cfg)
            # self.init_hgg_sampler(cfg)
       
            
        self.init_video_recorders(cfg)
        self.step = 0
        
        self.uniform_goal_sampler =  UniformFeasibleGoalSampler(env_name=cfg.env)
        self.init_dt_sampler()

    def init_dt_sampler(self):
        max_episode_timesteps = max_episode_timesteps_dict[self.cfg.env]
    
        hidden_size  = {'AntMazeSmall-v0' : 512,
                    'PointUMaze-v0' : 128,
                    'sawyer_peg_pick_and_place' : 256,
                    'sawyer_peg_push' : 256,
                    'PointNMaze-v0' : 128, 
                    'PointSpiralMaze-v0' : 256,
                    }[self.cfg.env]           

        lower_bounds, upper_bounds = self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS, self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS
        if lower_bounds.shape[0] == 2:
            limits = [[lower_bounds[0], upper_bounds[0]], [lower_bounds[1], upper_bounds[1]]]
        elif lower_bounds.shape[0] == 3:
            limits = [[lower_bounds[0], upper_bounds[0]], [lower_bounds[1], upper_bounds[1]], [lower_bounds[2], upper_bounds[2]]]
        else:
            raise NotImplementedError
        final_goal = get_original_final_goal(self.cfg.env)
        state_dim = final_goal.shape[0]

        dt = DecisionTransformer(state_dim = state_dim,
                                 act_dim= state_dim,
                                 max_length = max_episode_timesteps+10, 
                                 max_ep_len= max_episode_timesteps+10,
                                 hidden_size = hidden_size,
                                 n_layer = 3, # TODO check this out
                                 n_head = 1, #TODO check this out
                                 n_inner = 4* hidden_size,
                                 activation_func = 'relu', # TODO check out other loss functions that can be used
                                 n_positions = hidden_size * 4,
                                 resid_pdrop = 0.1,
                                 attn_pdrop = 0.1)
        
        optimizer = torch.optim.AdamW(
            dt.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )
        dt = dt.to(device=self.device)

        self.dt_sampler = DTSampler(self.env, self.eval_env, agent = self.get_agent(), optimizer= optimizer, dt= dt, video_recorder=self.train_video_recorder, env_name=self.cfg.env, max_ep_length = max_episode_timesteps+10, limits=limits, final_goal= final_goal)

    def init_env(self,cfg):
        cfg.max_episode_timesteps = max_episode_timesteps_dict[cfg.env]
        cfg.num_seed_steps = num_seed_steps_dict[cfg.env]
        cfg.num_random_steps = num_random_steps_dict[cfg.env]
        cfg.randomwalk_random_noise = randomwalk_random_noise_dict[cfg.env]
        assert cfg.aim_disc_replay_buffer_capacity == aim_disc_replay_buffer_capacity_dict[cfg.env]
     
        if cfg.env in ['sawyer_peg_push', 'sawyer_peg_pick_and_place']:
            cfg.goal_env=False
            from envs import sawyer_peg_pick_and_place, sawyer_peg_push
            if cfg.env =='sawyer_peg_pick_and_place':
                env = sawyer_peg_pick_and_place.SawyerPegPickAndPlaceV2(reward_type='sparse')
                eval_env = sawyer_peg_pick_and_place.SawyerPegPickAndPlaceV2(reward_type='sparse')
            elif cfg.env =='sawyer_peg_push':
                env = sawyer_peg_push.SawyerPegPushV2(reward_type='sparse', close_gripper=False)
                eval_env = sawyer_peg_push.SawyerPegPushV2(reward_type='sparse', close_gripper=False)
            
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env, max_episode_steps=cfg.max_episode_timesteps)
            eval_env = TimeLimit(eval_env, max_episode_steps=cfg.max_episode_timesteps)
            

            if cfg.use_residual_randomwalk:
                from env_utils import ResidualGoalWrapper
                env = ResidualGoalWrapper(env, env_name = cfg.env)
                eval_env = ResidualGoalWrapper(eval_env, env_name = cfg.env)

                                       
            from env_utils import StateWrapper, DoneOnSuccessWrapper
            if cfg.done_on_success:
                relative_goal_env = False
                residual_goal_env = True if cfg.use_residual_randomwalk else False
                env = DoneOnSuccessWrapper(env, relative_goal_env = (relative_goal_env or residual_goal_env), reward_offset=0.0, earl_env = False)
                eval_env = DoneOnSuccessWrapper(eval_env, relative_goal_env = (relative_goal_env or residual_goal_env), reward_offset=0.0, earl_env = False)

            from env_utils import WraptoGoalEnv
            self.env = StateWrapper(WraptoGoalEnv(env, env_name = cfg.env))
            self.eval_env = StateWrapper(WraptoGoalEnv(eval_env, env_name = cfg.env))

            obs_spec = self.env.observation_spec()
            action_spec = self.env.action_spec()
        
        elif cfg.goal_env: # e.g. Fetch, Ant
            import gym            
            from env_utils import StateWrapper, HERGoalEnvWrapper, DoneOnSuccessWrapper, ResidualGoalWrapper             
            if cfg.env in ['AntMazeSmall-v0']:
                from gym.wrappers.time_limit import TimeLimit
                from envs.AntEnv.envs.antenv import EnvWithGoal
                from envs.AntEnv.envs.antenv.create_maze_env import create_maze_env                                              
                self.env = TimeLimit(EnvWithGoal(create_maze_env(cfg.env, cfg.seed, env_path = cfg.env_path), cfg.env), max_episode_steps=cfg.max_episode_timesteps)
                self.eval_env = TimeLimit(EnvWithGoal(create_maze_env(cfg.env, cfg.seed, env_path = cfg.env_path), cfg.env), max_episode_steps=cfg.max_episode_timesteps)
                # self.eval_env.evaluate = True # set test goal = (0,16)
                self.env.set_attribute(evaluate=False, distance_threshold=1.0, horizon=cfg.max_episode_timesteps, early_stop=False)
                self.eval_env.set_attribute(evaluate=True, distance_threshold=1.0, horizon=cfg.max_episode_timesteps, early_stop=False)


                if cfg.use_residual_randomwalk:
                    self.env = ResidualGoalWrapper(self.env, env_name = cfg.env)
                    self.eval_env = ResidualGoalWrapper(self.eval_env, env_name = cfg.env)
            elif cfg.env in ["PointUMaze-v0", "PointSpiralMaze-v0", "PointNMaze-v0"]:
                from gym.wrappers.time_limit import TimeLimit
                import mujoco_maze                                             
                self.env = TimeLimit(gym.make(cfg.env), max_episode_steps=cfg.max_episode_timesteps)
                self.eval_env = TimeLimit(gym.make(cfg.env), max_episode_steps=cfg.max_episode_timesteps)

                if cfg.use_residual_randomwalk:
                    self.env = ResidualGoalWrapper(self.env, env_name = cfg.env)
                    self.eval_env = ResidualGoalWrapper(self.eval_env, env_name = cfg.env)

            else:
                self.env = gym.make(cfg.env)
                self.eval_env = gym.make(cfg.env)

            if cfg.done_on_success:
                relative_goal_env = False
                residual_goal_env = True if cfg.use_residual_randomwalk else False
                self.env = DoneOnSuccessWrapper(self.env, relative_goal_env = (relative_goal_env or residual_goal_env))
                self.eval_env = DoneOnSuccessWrapper(self.eval_env, relative_goal_env = (relative_goal_env or residual_goal_env))
            # self.goal_env = self.env
            
            self.env= StateWrapper(HERGoalEnvWrapper(self.env, env_name= cfg.env))
            self.eval_env= StateWrapper(HERGoalEnvWrapper(self.eval_env, env_name= cfg.env))
        
                
            obs_spec = self.env.observation_spec()
            action_spec = self.env.action_spec()
        return obs_spec, action_spec
    def set_cfg_agent(self, cfg, obs_spec, action_spec):
        cfg.agent.action_shape = action_spec.shape
        cfg.agent.action_range = [
            float(action_spec.low.min()),
            float(action_spec.high.max())
        ]
        cfg.agent.obs_shape = obs_spec.shape
        cfg.agent.goal_dim = self.env.goal_dim
    def init_hgg_achieved_trajectory_pool(self,cfg):
        from hgg.hgg import TrajectoryPool
        self.hgg_achieved_trajectory_pool = TrajectoryPool(**cfg.hgg_kwargs.trajectory_pool_kwargs)

    def init_hgg_sampler(self,cfg):
        from hgg.hgg import MatchSampler            
        self.hgg_sampler = MatchSampler(goal_env=self.eval_env, 
                                            goal_eval_env = self.eval_env, 
                                            env_name=cfg.env,
                                            achieved_trajectory_pool = self.hgg_achieved_trajectory_pool,
                                            agent = self.expl_agent,
                                            **cfg.hgg_kwargs.match_sampler_kwargs
                                            )      

    def init_feature_dim_for_discriminator_actor_critic(self,cfg):
        if cfg.env in ['sawyer_door', 'sawyer_peg']:      
            if cfg.aim_kwargs.aim_input_type=='default':
                cfg.aim_discriminator_cfg.x_dim = (get_object_states_only_from_goal(self.cfg.env, np.ones(self.env.goal_dim)).shape[-1])*2
            
            cfg.critic.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            cfg.actor.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            
        elif cfg.env in ['sawyer_peg_push', 'sawyer_peg_pick_and_place']:
            if cfg.aim_kwargs.aim_input_type=='default':
                cfg.aim_discriminator_cfg.x_dim = self.env.goal_dim*2
            
            cfg.critic.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            cfg.actor.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            
        else:
            if cfg.aim_kwargs.aim_input_type=='default':
                cfg.aim_discriminator_cfg.x_dim = self.env.goal_dim*2
            
            cfg.critic.feature_dim = self.env.obs_dim + self.env.goal_dim*2 # [obs, ag, dg]
            cfg.actor.feature_dim = self.env.obs_dim + self.env.goal_dim*2 # [obs, ag, dg]
            

        
    def init_video_recorders(self, cfg):
        self.eval_video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, dmc_env=False, env_name=cfg.env)
        self.train_video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, dmc_env=False, env_name=cfg.env)
        self.train_video_recorder.init(enabled=False)

    def init_buffers(self, cfg, obs_spec, action_spec):
        self.expl_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.replay_buffer_capacity,
                                        self.device)
        
        self.aim_expl_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.aim_disc_replay_buffer_capacity,
                                        self.device)
        n_sampled_goal = 4
        self.randomwalk_buffer = None
        if cfg.use_residual_randomwalk:
            self.randomwalk_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.randomwalk_buffer_capacity,
                                        self.device)
            self.randomwalk_buffer = HindsightExperienceReplayWrapperVer2(self.randomwalk_buffer, 
                                                                n_sampled_goal=n_sampled_goal, 
                                                                wrapped_env=self.env,
                                                                env_name = cfg.env,
                                                                consider_done_true = cfg.done_on_success,
                                                                )

        self.goal_buffer = None


        
        self.expl_buffer = HindsightExperienceReplayWrapperVer2(self.expl_buffer, 
                                                            n_sampled_goal=n_sampled_goal, 
                                                            wrapped_env=self.env,
                                                            env_name = cfg.env,
                                                            consider_done_true = cfg.done_on_success,
                                                            )
        # watch(self.expl_buffer.episode_transitions)
        self.aim_expl_buffer = HindsightExperienceReplayWrapperVer2(self.aim_expl_buffer, 
                                                            n_sampled_goal=cfg.aim_n_sampled_goal, #n_sampled_goal, 
                                                            # goal_selection_strategy=KEY_TO_GOAL_STRATEGY['future'],
                                                            wrapped_env=self.env,
                                                            env_name = cfg.env,
                                                            consider_done_true = cfg.done_on_success,
                                                            )

    def meta_nml_init(self, cfg): #no worries
        if cfg.use_meta_nml:
            if cfg.meta_nml.num_finetuning_layers in [None, 'none', 'None']:
                cfg.meta_nml.num_finetuning_layers = None
            if cfg.meta_nml_kwargs.meta_nml_custom_embedding_key in [None, 'none', 'None']:
                cfg.meta_nml_kwargs.meta_nml_custom_embedding_key = None
            
        cfg.meta_nml.equal_pos_neg_test= cfg.meta_nml_kwargs.equal_pos_neg_test and (not cfg.meta_nml_kwargs.meta_nml_negatives_only)
        cfg.meta_nml.input_dim = self.env.goal_dim


    def get_agent(self):                
        return self.expl_agent
        

    def get_buffer(self):                
        return self.expl_buffer


    def evaluate(self, eval_uniform_goal=False):
        uniform_goal=False
        repeat = 2 if eval_uniform_goal else 1
        visualizer_func = self.dt_sampler.visualizer.visualize_trajectories_on_time_on_eval
                        
        for r in range(repeat):            
            uniform_goal = True if r==1 else False

            avg_episode_reward = 0
            avg_episode_success_rate = 0       
            eval_subgoal_list = []
            for episode in range(self.cfg.num_eval_episodes):
                observes = []
                if uniform_goal:
                    sampled_goal = self.uniform_goal_sampler.sample()                    
                    obs = self.eval_env.reset(goal = sampled_goal)
                else:
                    obs = self.eval_env.reset()            
                
                final_goal = self.eval_env.goal.copy()                    
                        
                observes.append(obs)
                self.eval_video_recorder.init(enabled=False)
                episode_reward = 0
                episode_step = 0
                done = False
                while not done:                
                    agent = self.get_agent()
                    
                    with utils.eval_mode(agent):
                        action = agent.act(obs, spec = self.eval_env.action_spec(), sample=False)
                    next_obs, reward, done, info = self.eval_env.step(action)
                    self.eval_video_recorder.record(self.eval_env)
                    episode_reward += reward
                    episode_step += 1
                    obs = next_obs
                    
                    if self.cfg.use_residual_randomwalk:
                        if ((episode_step) % self.max_episode_timesteps == 0) or info.get('is_current_goal_success'):
                            done = True

                    observes.append(obs)
                    

            
                observes = np.stack(observes, axis =0)
                obs_dict = self.eval_env.convert_obs_to_dict(observes)
                visualizer_func(self.episode, episode, obs_dict['achieved_goal'])
                if self.eval_env.is_successful(obs):
                    avg_episode_success_rate+=1.0
                
                if self.cfg.use_aim and episode==0:                
                    fig = plt.figure(figsize=(15,15))
                    sns.set_style("darkgrid")

                    tiled_initial_obs = np.tile(obs_dict['achieved_goal'][0][None, :], (observes.shape[0], 1)) #[ts, dim]
                    
                    obs_desired_goal = obs_dict['desired_goal']                
                        

                    if self.cfg.env in ['sawyer_door', 'sawyer_peg']:
                        if self.cfg.aim_kwargs.aim_input_type=='default':
                            observes = torch.from_numpy(np.concatenate([get_object_states_only_from_goal(self.cfg.env, obs_dict['achieved_goal']), get_object_states_only_from_goal(self.cfg.env, obs_desired_goal)], axis =-1)).float().to(self.device) #[ts, dim]
                            observes_reverse = torch.from_numpy(np.concatenate([get_object_states_only_from_goal(self.cfg.env, obs_dict['achieved_goal']), get_object_states_only_from_goal(self.cfg.env, tiled_initial_obs)], axis =-1)).float().to(self.device) #[ts, dim]

                    else:
                        if self.cfg.aim_kwargs.aim_input_type=='default':
                            observes = torch.from_numpy(np.concatenate([obs_dict['achieved_goal'], obs_desired_goal], axis =-1)).float().to(self.device) #[ts, dim]
                            observes_reverse = torch.from_numpy(np.concatenate([obs_dict['achieved_goal'], tiled_initial_obs], axis =-1)).float().to(self.device) #[ts, dim]


                    
                    aim_reward = self.expl_agent.compute_aim_reward(observes).detach().cpu().numpy()
                    if self.cfg.normalize_f_obs:
                        aim_disc_outputs = self.expl_agent.aim_discriminator.forward(self.expl_agent.normalize_obs(observes, self.cfg.env)).detach().cpu().numpy()
                    else:
                        aim_disc_outputs = self.expl_agent.aim_discriminator.forward(observes).detach().cpu().numpy()
                    
                    aim_reward_reverse = self.expl_agent.compute_aim_reward(observes_reverse).detach().cpu().numpy()
                    if self.cfg.normalize_f_obs:
                        aim_disc_outputs_reverse = self.expl_agent.aim_discriminator.forward(self.expl_agent.normalize_obs(observes_reverse, self.cfg.env)).detach().cpu().numpy()
                    else:
                        aim_disc_outputs_reverse = self.expl_agent.aim_discriminator.forward(observes_reverse).detach().cpu().numpy()

                    
                    timesteps = np.arange(observes.shape[0])
                    ax1 = fig.add_subplot(4,1,1)                
                    ax1.plot(timesteps, aim_reward, label = 'aim_reward')
                    ax1.legend(loc ='upper right') # , prop={'size': 20}
                    ax2 = fig.add_subplot(4,1,2)
                    ax2.plot(timesteps, aim_disc_outputs, label = 'aim_disc_output')
                    ax2.legend(loc ='upper right')
                    ax3 = fig.add_subplot(4,1,3)                
                    ax3.plot(timesteps, aim_reward_reverse, label = 'aim_reward_reverse')
                    ax3.legend(loc ='upper right') # , prop={'size': 20}
                    ax4 = fig.add_subplot(4,1,4)
                    ax4.plot(timesteps, aim_disc_outputs_reverse, label = 'aim_disc_output_reverse')
                    ax4.legend(loc ='upper right')
                    if uniform_goal:
                        plt.savefig(self.eval_video_recorder.save_dir+'/aim_outputs_uniform_goal_'+str(self.step)+'.jpg')
                    else:
                        plt.savefig(self.eval_video_recorder.save_dir+'/aim_outputs_'+str(self.step)+'.jpg')
                    plt.close()
                    
                avg_episode_reward += episode_reward
                if uniform_goal:
                    self.eval_video_recorder.save(f'uniform_goal_{self.step}.mp4')
                else:
                    self.eval_video_recorder.save(f'{self.step}.mp4')
            avg_episode_reward /= self.cfg.num_eval_episodes
            avg_episode_success_rate = avg_episode_success_rate/self.cfg.num_eval_episodes
            if uniform_goal:                
                self.eval_env.reset(goal = get_original_final_goal(self.cfg.env))
                self.logger.log('eval/episode_reward_uniform_goal', avg_episode_reward, self.step)
                self.logger.log('eval/episode_success_rate_uniform_goal', avg_episode_success_rate, self.step)
            else:
                self.logger.log('eval/episode_reward', avg_episode_reward, self.step)
                self.logger.log('eval/episode_success_rate', avg_episode_success_rate, self.step)

                

        self.logger.dump(self.step, ty='eval')
            
    
    def get_inv_weight_curriculum_buffer(self):        

        if self.cfg.inv_weight_curriculum_kwargs.curriculum_buffer=='aim':
            return self.aim_expl_buffer
        elif self.cfg.inv_weight_curriculum_kwargs.curriculum_buffer=='default':
            return self.expl_buffer

    def run(self):        
        self._run()

    def dt_sampler_update(self,step, episode, episode_observes, episode_acts, qs):
        self.dt_sampler.update(step, episode, episode_observes, episode_acts, qs)

    def _run(self):        
        episode, episode_reward, episode_step, start_time, recent_sampled_goals, done, info, current_pocket_success, current_pocket_trial = self.run_init()
        agent = self.get_agent()
        first_iteration = True
        
        while self.step <= self.cfg.num_train_steps:
            
            if done:
                if self.step > 0:
                    current_pocket_trial += 1
                    if info['is_success']:
                        current_pocket_success += 1 
                    
                    # # hgg update
                    # if self.cfg.use_hgg :
                    #     if episode % self.cfg.hgg_kwargs.hgg_sampler_update_frequency ==0 :                            
                    #         self.hgg_update(episode)
                




                self.train_video_recorder.save(f'train_episode_{episode-1}.mp4')                
                if self.step > 0: #about logging
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()
                    self.logger.log('train/episode_reward', episode_reward, self.step)
                    self.logger.log('train/episode', episode, self.step)
                
                # obs = self.hgg_sample(recent_sampled_goals)
                obs = self.dt_sample(recent_sampled_goals)
                residual_goals_observes = []
                final_goal = self.env.goal.copy()                
                
                    
                self.logger.log('train/episode_finalgoal_dist', np.linalg.norm(final_goal), self.step)
                if self.cfg.use_hgg:
                    original_final_goal = get_original_final_goal(self.cfg.env)                    
                    self.logger.log('train/episode_dist_from_curr_g_to_example_g', np.linalg.norm(final_goal-original_final_goal), self.step)                                        
                    sampled_goals_for_log = np.array(recent_sampled_goals.queue)
                    self.logger.log('train/average_dist_from_curr_g_to_example_g', np.linalg.norm(original_final_goal[None, :]-sampled_goals_for_log, axis =-1).mean(), self.step)
                        


                self.visualize_training(episode, recent_sampled_goals, agent, obs)


                episode_reward = 0
                episode_step = 0
                episode += 1
                episode_observes = []
                episode_acts     = []
                qs = np.array([]).reshape(0,2)

                self.logger.log('train/episode', episode, self.step)

            replay_buffer = self.get_buffer()
            # evaluate agent periodically and visualize
            if self.step % self.cfg.eval_frequency == 0:
                print('eval started...')
                self.logger.log('eval/episode', episode - 1, self.step)
                self.evaluate(eval_uniform_goal=False)                

                if self.step > self.cfg.num_random_steps:
                    self.visualize_curriculum_goals(agent)
                
                if self.cfg.use_residual_randomwalk and (self.randomwalk_buffer.idx > 128 or self.randomwalk_buffer.full):
                    self.visualize_residual_walk_and_goals(agent)
                    

            
            self.periodic_save()  
            action = self.get_agent_act(obs)
            logging_dict = agent.update(replay_buffer, self.randomwalk_buffer, self.aim_expl_buffer, self.step, self.env, self.goal_buffer)
            

            qs = np.vstack((qs ,[logging_dict.get('q1', 0), logging_dict.get('q2', 0)]))         
            if self.step % self.cfg.logging_frequency== 0:                
                if logging_dict is not None: # when step = 0                                        
                    for key, val in logging_dict.items():
                        self.logger.log('train/'+key, val, self.step)
            
           
            next_obs, reward, done, info = self.env.step(action)
            # self.env.render()
            episode_reward += reward
            episode_observes.append(obs)
            episode_acts.append(action)

            last_timestep = True if (episode_step+1) % self.max_episode_timesteps == 0 or done else False


            
            self.train_video_recorder.record(self.env)


            
            self.buffer_update(done, info, obs, replay_buffer, action, next_obs, reward, last_timestep)
            
                
            if last_timestep:
                self.last_timestep_save(episode_observes, replay_buffer)
                qs = np.array(qs, dtype = float)
                self.dt_sampler_update(self.step, episode, episode_observes, episode_acts, qs)
                self.episode = episode

                
                    
                    
                    

            obs = next_obs
            episode_step += 1
            self.step += 1
            
            if self.cfg.use_residual_randomwalk:
                if self.env.is_residual_goal:
                    if info.get('is_current_goal_success'):
                        if (self.cfg.use_uncertainty_for_randomwalk not in [None, 'none', 'None']) and self.step > self.get_agent().meta_test_sample_size:
                            # residual_goal = self.get_residual_goal_with_aim(episode, obs)
                            # qs = np.array(qs, dtype = float)
                            residual_goal = self.get_residual_goal_with_dt(episode,episode_observes, episode_acts, qs)
                            if residual_goal is None:
                                residual_goal = self.get_residual_goal_with_random(obs)
                            # residual_goal = self.get_residual_goal_with_nonNML(episode, obs)

                        else:
                            # residual_goal = self.get_residual_goal_with_random(obs)
                            # qs = np.array(qs, dtype = float)

                            residual_goal = self.get_residual_goal_with_dt(episode,episode_observes, episode_acts, qs)
                            if residual_goal is None:
                                residual_goal = self.get_residual_goal_with_random(obs)
                            # residual_goal = self.get_residual_goal_with_nonNML(episode, obs)
                        self.env.reset_goal(residual_goal)
                        obs[-self.env.goal_dim:] = residual_goal.copy()
                else:
                    if info.get('is_current_goal_success'): #succeed original goal
                        self.env.original_goal_success = True
                        if (self.cfg.use_uncertainty_for_randomwalk not in [None, 'none', 'None']) and self.step > self.get_agent().meta_test_sample_size:
                            # qs = np.array(qs, dtype = float)
                            residual_goal = self.get_residual_goal_with_dt(episode,episode_observes, episode_acts, qs)
                            if residual_goal is None:
                                residual_goal = self.get_residual_goal_with_random(obs)                            # residual_goal = self.get_residual_goal_with_random(obs)

                            # residual_goal = self.get_residual_goal_with_aim(episode, obs)
                        else:
                            # qs = np.array(qs, dtype = float)
                            residual_goal = self.get_residual_goal_with_dt(episode,episode_observes, episode_acts, qs)
                            if residual_goal is None:
                                residual_goal = self.get_residual_goal_with_random(obs)                            # residual_goal = self.get_residual_goal_with_random(obs)



                        self.env.reset_goal(residual_goal)
                        obs[-self.env.goal_dim:] = residual_goal.copy()
                if (episode_step) % self.max_episode_timesteps == 0: #done only horizon ends
                    done = True
                    info['is_success'] = self.env.original_goal_success

    def buffer_update(self, done, info, obs, replay_buffer, action, next_obs, reward, last_timestep):
        if self.cfg.use_residual_randomwalk:
            if self.env.is_residual_goal:
                self.randomwalk_buffer.add(obs, action, reward, next_obs, info.get('is_current_goal_success'), last_timestep)
            else:
                replay_buffer.add(obs, action, reward, next_obs, info.get('is_current_goal_success'), last_timestep)
                self.aim_expl_buffer.add(obs, action, reward, next_obs, info.get('is_current_goal_success'), last_timestep)

        else:
            replay_buffer.add(obs, action, reward, next_obs, done, last_timestep)
            self.aim_expl_buffer.add(obs, action, reward, next_obs, done, last_timestep)

    def get_residual_goal_with_random(self, obs):
        noise = np.random.uniform(low=-self.cfg.randomwalk_random_noise, high=self.cfg.randomwalk_random_noise, size=self.env.goal_dim)
                            
        if self.cfg.env in [   'sawyer_peg_pick_and_place']:
            assert self.cfg.randomwalk_random_noise <= 0.2
            pass
        elif self.cfg.env in [  'sawyer_peg_push']:
            assert self.cfg.randomwalk_random_noise <= 0.2
            noise[2] = 0
        residual_goal = self.env.convert_obs_to_dict(obs)['achieved_goal'] + noise
        return residual_goal

    def get_residual_goal_with_aim(self, episode, obs):
        return self.get_agent().sample_randomwalk_goals(obs = obs, ag = self.env.convert_obs_to_dict(obs)['achieved_goal'], \
                                episode = episode, env=self.env, replay_buffer = self.get_inv_weight_curriculum_buffer(), \
                                num_candidate = self.cfg.randomwalk_num_candidate, random_noise = self.cfg.randomwalk_random_noise, \
                                uncertainty_mode = self.cfg.use_uncertainty_for_randomwalk)
    def get_residual_goal_with_dt(self, episode, episode_observes,episode_acts, qs):
        return self.dt_sampler.sample(episode_observes, episode_acts, qs)

    def last_timestep_save(self, episode_observes, replay_buffer):
        replay_buffer.add_trajectory(episode_observes)
        replay_buffer.store_episode()
        self.aim_expl_buffer.store_episode()
        if self.randomwalk_buffer is not None:
            self.randomwalk_buffer.store_episode()
        if self.randomwalk_buffer is not None:
            if (not replay_buffer.full) and (not self.randomwalk_buffer.full):
                assert self.step+1 == self.randomwalk_buffer.idx + replay_buffer.idx
        else:
            if not replay_buffer.full:
                assert self.step+1 == replay_buffer.idx

        if self.cfg.use_hgg:                    
            temp_episode_observes = copy.deepcopy(episode_observes)
            temp_episode_ag = []                                        
                     # NOTE : should it be [obs, ag] ?
            temp_episode_init = self.eval_env.convert_obs_to_dict(temp_episode_observes[0])['achieved_goal'] # for bias computing

                        

            for k in range(len(temp_episode_observes)):
                temp_episode_ag.append(self.eval_env.convert_obs_to_dict(temp_episode_observes[k])['achieved_goal'])
                    
            if getattr(self.env, 'full_state_goal', False):
                raise NotImplementedError("You should modify the code when full_state_goal (should address achieved_goal to compute goal distance below)")


            achieved_trajectories = [np.array(temp_episode_ag)] # list of [ts, dim]
            achieved_init_states = [temp_episode_init] # list of [ts(1), dim]

            selection_trajectory_idx = {}
            for i in range(len(achieved_trajectories)):                                                 
                        # full state achieved_goal
                if self.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0", "PointSpiralMaze-v0", "PointNMaze-v0"]:
                    threshold = 0.2
                elif self.cfg.env in [     'sawyer_peg_push','sawyer_peg_pick_and_place']:
                    threshold = 0.02
                else:
                    raise NotImplementedError
                if goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>threshold: # if there is a difference btw first and last timestep ?
                    selection_trajectory_idx[i] = True
                    
            hgg_achieved_trajectory_pool = self.hgg_achieved_trajectory_pool
            for idx in selection_trajectory_idx.keys():
                hgg_achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())
                
    def get_agent_act(self,obs):
            agent = self.get_agent()
            # sample action for data collection
            if self.step < self.cfg.num_random_steps or (self.cfg.randomwalk_method == 'rand_action' and self.env.is_residual_goal):
                spec = self.env.action_spec()                
                action = np.random.uniform(spec.low, spec.high,
                                        spec.shape)
                
            else: 
                with utils.eval_mode(agent):
                    action = agent.act(obs, spec = self.env.action_spec(), sample=True)
            return action
    def periodic_save(self):
        if self.cfg.save_model and self.step % self.cfg.save_frequency == 0:
            utils.save(
                self.expl_agent,
                os.path.join(self.model_dir, f'expl_agent_{self.step}.pt'))                
        if self.cfg.save_buffer and (self.step % self.cfg.buffer_save_frequency == 0) :
            utils.save(self.expl_buffer.replay_buffer, os.path.join(self.buffer_dir, f'buffer_{self.step}.pt'))
            utils.save(self.aim_expl_buffer.replay_buffer, os.path.join(self.buffer_dir, f'aim_disc_buffer_{self.step}.pt'))
            if self.cfg.use_residual_randomwalk:
                utils.save(self.randomwalk_buffer.replay_buffer, os.path.join(self.buffer_dir, f'randomwalk_buffer_{self.step}.pt'))
        
            if self.cfg.use_hgg:
                utils.save(self.hgg_achieved_trajectory_pool,  os.path.join(self.buffer_dir, f'hgg_achieved_trajectory_pool_{self.step}.pt'))

    def visualize_residual_walk_and_goals(self, agent):
        temp_obs, _, _, _, _, _ = self.randomwalk_buffer.sample_without_relabeling(128, agent.discount, sample_only_state = False)
        temp_obs = temp_obs.detach().cpu().numpy()
        temp_obs_dict = self.env.convert_obs_to_dict(temp_obs)
                     
        temp_dg = temp_obs_dict['desired_goal']
        temp_ag = temp_obs_dict['achieved_goal']
                    
        fig = plt.figure()
        sns.set_style("darkgrid")
                    
        ax1 = fig.add_subplot(1,1,1)                                    
        ax1.scatter(temp_dg[:, 0], temp_dg[:, 1], label = 'goals')
        ax1.scatter(temp_ag[:, 0], temp_ag[:, 1], label = 'achieved states', color = 'red')
                              
        if self.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0"]:
            x_min, x_max = -2, 10
            y_min, y_max = -2, 10
        elif self.cfg.env == "PointSpiralMaze-v0":
            x_min, x_max = -10, 10
            y_min, y_max = -10, 10
        elif self.cfg.env in ["PointNMaze-v0"]:
            x_min, x_max = -2, 10
            y_min, y_max = -2, 18
        elif self.cfg.env in [     'sawyer_peg_push','sawyer_peg_pick_and_place']:
            x_min, x_max = -0.6, 0.6
            y_min, y_max = 0.2, 1.0
        else:
            raise NotImplementedError
        plt.xlim(x_min,x_max)    
        plt.ylim(y_min,y_max)
                    
                    

        ax1.legend(loc ="best") # 'upper right' # , prop={'size': 20}          
        plt.savefig(self.eval_video_recorder.save_dir+'/randomwalk_goalandstates_'+str(self.step)+'.jpg')
        plt.close()

    def visualize_curriculum_goals(self, agent):
        temp_obs, _, _, _, _, _ = self.aim_expl_buffer.sample_without_relabeling(128, agent.discount, sample_only_state = False)
        temp_obs = temp_obs.detach().cpu().numpy()
        temp_obs_dict = self.env.convert_obs_to_dict(temp_obs)
                     
        temp_dg = temp_obs_dict['desired_goal']
        temp_ag = temp_obs_dict['achieved_goal']

        fig = plt.figure()
        sns.set_style("darkgrid")
                    
        ax1 = fig.add_subplot(1,1,1)                                    
        ax1.scatter(temp_dg[:, 0], temp_dg[:, 1], label = 'desired goals', color = 'red')
        ax1.scatter(temp_ag[:, 0], temp_ag[:, 1], label = 'achieved goals', color = 'green')
                              
        if self.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0"]:
            x_min, x_max = -2, 10
            y_min, y_max = -2, 10
        elif self.cfg.env == "PointSpiralMaze-v0":
            x_min, x_max = -10, 10
            y_min, y_max = -10, 10
        elif self.cfg.env in ["PointNMaze-v0"]:
            x_min, x_max = -2, 10
            y_min, y_max = -2, 18
        elif self.cfg.env in [     'sawyer_peg_push','sawyer_peg_pick_and_place']:
            x_min, x_max = -0.6, 0.6
            y_min, y_max = 0.2, 1.0
        else:
            raise NotImplementedError

        plt.xlim(x_min,x_max)    
        plt.ylim(y_min,y_max)
                    
                                  

        ax1.legend(loc ="best") # 'upper right' # , prop={'size': 20}          
        plt.savefig(self.eval_video_recorder.save_dir+'/curriculum_goals_'+str(self.step)+'.jpg')
        plt.close()

    def visualize_training(self, episode, recent_sampled_goals, agent, obs):
        self.train_video_recorder.init(enabled=False)
                
        hgg_save_freq = 3 if 'Point' in self.cfg.env else 25
        if self.cfg.use_hgg and episode % hgg_save_freq == 0 :
            sampled_goals_for_vis = np.array(recent_sampled_goals.queue) 
            fig = plt.figure()
            sns.set_style("darkgrid")
            ax1 = fig.add_subplot(1,1,1)                    
            ax1.scatter(sampled_goals_for_vis[:, 0], sampled_goals_for_vis[:, 1])
            if self.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0"]:
                plt.xlim(-2,10)    
                plt.ylim(-2,10)
            elif self.cfg.env == "PointSpiralMaze-v0":
                plt.xlim(-10,10)    
                plt.ylim(-10,10)
            elif self.cfg.env in ["PointNMaze-v0"]:
                plt.xlim(-2,10)    
                plt.ylim(-2,18)
            elif self.cfg.env in [     'sawyer_peg_push','sawyer_peg_pick_and_place']:
                plt.xlim(-0.6,0.6)    
                plt.ylim(0.2,1.0)
            else:
                raise NotImplementedError
            plt.savefig(self.train_video_recorder.save_dir+'/train_hgg_goals_episode_'+str(episode)+'.jpg')
            plt.close()
            with open(self.train_video_recorder.save_dir+'/train_hgg_goals_episode_'+str(episode)+'.pkl', 'wb') as f:
                pkl.dump(sampled_goals_for_vis, f)


        if episode % self.cfg.train_episode_video_freq == 0 or episode in [25,50,75,100]:             
            self.train_video_recorder.init(enabled=False)           
                    # Visualize from init state to subgoals
                   

            visualize_num_iter = 0
            scatter_states = self.env.convert_obs_to_dict(obs.copy())['achieved_goal'][None, :]
                    
            for k in range(visualize_num_iter+1):                                                
                init_state = scatter_states[k]
                if self.cfg.use_aim:
                    visualize_discriminator(normalizer = agent.normalize_obs if self.cfg.normalize_f_obs else None,
                                                    discriminator = agent.aim_discriminator, 
                                                    initial_state = init_state, 
                                                    scatter_states = scatter_states.squeeze(),
                                                    env_name = self.cfg.env, 
                                                    aim_input_type = self.cfg.aim_kwargs.aim_input_type,
                                                    device = self.device, 
                                                    savedir_w_name = self.train_video_recorder.save_dir + '/aim_f_visualize_train_episode_'+str(episode)+'_s'+str(k),
                                                    )

                    visualize_discriminator2(normalizer = agent.normalize_obs if self.cfg.normalize_f_obs else None,
                                                    discriminator = agent.aim_discriminator, 
                                                    env_name = self.cfg.env, 
                                                    aim_input_type = self.cfg.aim_kwargs.aim_input_type,
                                                    device = self.device, 
                                                    savedir_w_name = self.train_video_recorder.save_dir + '/aim_f_visualize_train_goalfix_'+str(episode)+'_s'+str(k),
                                                    )
                # if self.cfg.use_meta_nml:
                #     visualize_meta_nml(agent=agent, 
                #                                meta_nml_epoch=episode, 
                #                                scatter_states = scatter_states.squeeze(),
                #                                replay_buffer= self.get_buffer(), 
                #                                goal_env = self.env,
                #                                env_name = self.cfg.env, 
                #                                aim_input_type = self.cfg.aim_kwargs.aim_input_type, 
                #                                savedir_w_name = self.train_video_recorder.save_dir + '/aim_meta_nml_prob_visualize_train_episode_'+str(episode)+'_s'+str(k),
                #                                )

    def hgg_sample(self, recent_sampled_goals):
        obs = None
        if self.cfg.use_hgg:                    
            hgg_sampler = self.hgg_sampler
            n_iter = 0
            while True:
                        # print('hgg sampler pool len : {} step : {}'.format(len(hgg_sampler.pool), self.step))
                sampled_goal = hgg_sampler.sample(np.random.randint(len(hgg_sampler.pool))).copy()   # curriculum i cikarir                     
                obs = self.env.reset(goal = sampled_goal)

                if not self.env.is_successful(obs):
                    break
                n_iter +=1
                if n_iter==10:
                    break

            if recent_sampled_goals.full():
                recent_sampled_goals.get()
            recent_sampled_goals.put(sampled_goal)
                    # obs = self.env.reset(goal = sampled_goal)
            assert (sampled_goal == self.env.goal.copy()).all()
        else:
            obs = self.env.reset()
        return obs
    def dt_sample(self, recent_sampled_goals):
        obs = None
        dt_hgg_sampler = self.dt_sampler
        n_iter = 0 
        while True:
            sampled_goal = dt_hgg_sampler.sample()
            obs = self.env.reset(goal = sampled_goal)

            if not self.env.is_successful(obs):
                break
            n_iter +=1
            if n_iter == 10:
                break
        if recent_sampled_goals.full():
            recent_sampled_goals.get()
        recent_sampled_goals.put(sampled_goal)

        return obs

        
    def run_init(self):
        episode, episode_reward, episode_step = 0, 0, 0
        inv_curriculum_pocket = []
        start_time = time.time()

        if self.cfg.use_hgg:
            recent_sampled_goals = Queue(self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes)
        


        previous_goals = None
        done = True
        info = {}

        
        self.meta_nml_run_init()

        if self.cfg.use_hgg:
            temp_obs = self.eval_env.reset()        # reset everything, and get the observations of the moment
            recent_sampled_goals.put(self.eval_env.convert_obs_to_dict(temp_obs)['achieved_goal'].copy()) #add the first observation after reset


        current_pocket_success = 0
        current_pocket_trial = 0
        return episode,episode_reward,episode_step,start_time,recent_sampled_goals,done,info,current_pocket_success,current_pocket_trial

    def meta_nml_run_init(self):
        if self.cfg.use_meta_nml:                
            if self.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0"]:  
                final_goal_states = np.tile(np.array([0., 8.]), (self.cfg.aim_num_precollect_init_state,1))
                final_goal_states += np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.5*np.ones_like(final_goal_states))
            elif self.cfg.env  == "PointSpiralMaze-v0":
                final_goal_states = np.tile(np.array([8., -8.]), (self.cfg.aim_num_precollect_init_state,1))
                final_goal_states += np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.5*np.ones_like(final_goal_states))
            elif self.cfg.env in ["PointNMaze-v0"]:
                final_goal_states = np.tile(np.array([8., 16.]), (self.cfg.aim_num_precollect_init_state,1))
                final_goal_states += np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.5*np.ones_like(final_goal_states))
            elif self.cfg.env in [    'sawyer_peg_push' ]:
                final_goal_states = np.tile(np.array([-0.3, 0.4, 0.02]), (self.cfg.aim_num_precollect_init_state,1))
                noise = np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.05*np.ones_like(final_goal_states))
                noise[2] = 0
                final_goal_states += noise
            elif self.cfg.env in ['sawyer_peg_pick_and_place']:
                final_goal_states = np.tile(np.array([-0.3, 0.4, 0.2]), (self.cfg.aim_num_precollect_init_state,1))
                final_goal_states += np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.05*np.ones_like(final_goal_states))
            else:
                raise NotImplementedError
            self.get_agent().final_goal_states = final_goal_states.copy()
            


                    

@hydra.main(config_path='./config', config_name='config_outpace.yaml')
def main(cfg):
    import os
    os.environ['HYDRA_FULL_ERROR'] = str(1)
    from outpace_train import Workspace as W
    

    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()

