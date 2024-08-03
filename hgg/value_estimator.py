import torch 
import numpy as np
from playground2 import time_decorator
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
class ValueEstimator:
    def __init__(self, agent, eval_env, final_goal, num_seed_steps, init_goal, gamma, beta, sigma):
        self.agent = agent
        self.eval_env = eval_env
        self.final_goal = final_goal
        self.num_seed_steps = num_seed_steps
        self.step = 0
        self.init_goal = init_goal
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma
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

        return values
    def generate_achieved_values_t(self,init_state,achieved_goals):

        goal_pairs = [goal_concat(init_state, achieved_goals[j]) for j in range(len(achieved_goals))]
        init_to_goal_pair = torch.stack(goal_pairs).to(device='cuda', dtype=torch.float32)

        if not isinstance(init_to_goal_pair, torch.Tensor):
            init_to_goal_pair = torch.tensor(init_to_goal_pair, device = "cuda", dtype = torch.float32)

        goal_to_final_pair = [goal_concat(achieved_goals[j], self.final_goal) for j in range(len(achieved_goals))]
        if not isinstance(goal_to_final_pair[0], torch.Tensor):  # Check if the elements are tensors
            goal_to_final_pair = torch.tensor(goal_to_final_pair, device="cuda", dtype=torch.float32)
        else:
            goal_to_final_pair = torch.stack(goal_to_final_pair).to(device="cuda", dtype=torch.float32)

        values = -self.agent.aim_discriminator(init_to_goal_pair)[:, 0] # TODO discover inside aim_discriminator,
                                                                                            # 	* what kind of inputs it does require
                                                                                            # 	* value can be interpreted as a measure of how desirable or advantageous the current state is from the perspective of achieving the final goal

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

        return q1,q2

    def calculate_exploration_value(self, init_pos, curr_pos):
        epsilon = 1e-10  # Small value to prevent division by zero
        if type(init_pos) == torch.Tensor:
            numerator = torch.linalg.norm(curr_pos - init_pos)
            denominator = torch.linalg.norm(torch.tensor(self.final_goal, device= "cuda") - curr_pos) + epsilon
            value = torch.log(numerator + epsilon) - torch.log(denominator)

            # value = numerator / denominator
            return numerator-denominator	
        else:
            numerator = np.linalg.norm(curr_pos - init_pos)
            denominator = np.linalg.norm(self.final_goal - curr_pos) + epsilon
            value = np.log(numerator + epsilon) - np.log(denominator)

            # value = numerator / denominator
            return numerator-denominator
        
    def get_state_values_t(self, achieved_goals):
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
        q_values_t = q_values_t.squeeze(1).squeeze(1)
        state_values_t = self.gamma * achieved_values_t + self.beta * q_values_t + self.sigma * exploration_values_t
        state_values_t = state_values_t.unsqueeze(0)
        return state_values_t, achieved_values_t, exploration_values_t, q_values_t
    
    def get_state_values(self,achieved_goals, qs):
        achieved_values= self.generate_achieved_values(self.init_goal,achieved_goals)
        exploration_values = np.array([self.calculate_exploration_value(self.init_goal,achieved_goals[i]) for i in range(len(achieved_goals))])

        if qs is None:
            qs = np.array([])
            for pos in achieved_goals:
                q1_t, q2_t = self.get_q_values(torch.tensor(pos, device="cuda", dtype=torch.float32))

                q_min_t = torch.min(q1_t,q2_t)
                q_min = q_min_t.detach().cpu().numpy()
                qs = np.append(qs,q_min)
        else:
            qs = np.min(qs, axis= 1)
        total_val = self.gamma * achieved_values + self.beta * qs + self.sigma * exploration_values
        return total_val, achieved_values, exploration_values, qs
    


    def get_max_min_rewards(self, init_state):
        q1_t, q2_t = self.value_estimator.get_q_values(torch.tensor(self.final_goal, device = "cuda", dtype = torch.float32))
        q_val = np.min([q1_t.detach().cpu().numpy(), q2_t.detach().cpu().numpy()])
        aim_val = self.value_estimator.generate_achieved_values(init_state, [self.final_goal])[0]
        expl_val = self.value_estimator.calculate_exploration_value(init_state, self.final_goal)

        max_aim 	= aim_val
        max_expl	= expl_val
        max_q		= q_val

        q1_t, q2_t = self.value_estimator.get_q_values(torch.tensor(init_state, device = "cuda", dtype = torch.float32))
        q_val = np.min([q1_t.detach().cpu().numpy(), q2_t.detach().cpu().numpy()])
        aim_val = self.value_estimator.generate_achieved_values(init_state, [init_state])[0]
        expl_val = self.value_estimator.calculate_exploration_value(init_state, init_state)

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