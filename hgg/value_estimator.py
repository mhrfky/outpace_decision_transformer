import torch 
import numpy as np
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
    def __init__(self, agent, eval_env, final_goal, num_seed_steps, init_goal, gamma, beta, sigma, rescale = False):
        self.agent = agent
        self.eval_env = eval_env
        self.final_goal = final_goal
        self.num_seed_steps = num_seed_steps
        self.step = 0
        self.init_goal = np.array([0,0])
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma
        self.rescale = rescale

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
            numerator = torch.linalg.norm(curr_pos - init_pos) + epsilon
            denominator = torch.linalg.norm(torch.tensor(self.final_goal, device= "cuda") - curr_pos) + epsilon
            value = torch.log(numerator + epsilon) - torch.log(denominator)

            # value = numerator / denominator
            return value # numerator-denominator	
        else:
            numerator = np.linalg.norm(curr_pos - init_pos) + epsilon
            denominator = np.linalg.norm(self.final_goal - curr_pos) + epsilon
            value = np.log(numerator + epsilon) - np.log(denominator)

            # value = numerator / denominator
            return value #numerator ** 2 - denominator ** 2
        
    def get_scale_min_max(self):

        q_min = self.get_q_values(torch.tensor(self.init_goal, device="cuda", dtype=torch.float32))[0].item()
        q_max = self.get_q_values(torch.tensor(self.final_goal, device="cuda", dtype=torch.float32))[0].item()

        # aim_min = self.generate_achieved_values(self.init_goal, [self.init_goal])[0]
        # aim_max = self.generate_achieved_values(self.init_goal, [self.final_goal])[0]

        expl_min= self.calculate_exploration_value(self.init_goal, self.init_goal)
        expl_max= self.calculate_exploration_value(self.init_goal, self.final_goal)

        return q_min, q_max, 0, 0, expl_min, expl_max
    
    def get_scale_min_max_t(self):
        q_min = self.get_q_values(torch.tensor(self.init_goal, device="cuda", dtype=torch.float32))[0].item()
        q_max = self.get_q_values(torch.tensor(self.final_goal, device="cuda", dtype=torch.float32))[0].item()

        # aim_min = self.generate_achieved_values_t(self.init_goal, [self.init_goal])[0]
        # aim_max = self.generate_achieved_values_t(self.init_goal, [self.final_goal])[0]

        init_goal_t = torch.tensor(self.init_goal, device="cuda", dtype=torch.float32)
        final_goal_t = torch.tensor(self.final_goal, device="cuda", dtype=torch.float32)
        expl_min= self.calculate_exploration_value(init_goal_t, init_goal_t)
        expl_max= self.calculate_exploration_value(init_goal_t, final_goal_t)

        return q_min, q_max, 0, 0, expl_min, expl_max
    
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
        if self.rescale:
            q_min, q_max, aim_min, aim_max, expl_min, expl_max = self.get_scale_min_max_t()
            q_values_t = self.rescale_values(q_values_t, q_min, q_max)
            # achieved_values_t = self.rescale_values(achieved_values_t, aim_min, aim_max)
            exploration_values_t = (exploration_values_t - expl_min) / (expl_max - expl_min)

        return state_values_t, achieved_values_t, exploration_values_t, q_values_t
    def rescale_values(self, values, min_val, max_val):
           values -= min_val
           values /= (max_val - min_val)
           return values
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
        if self.rescale:
            q_min, q_max, aim_min, aim_max, expl_min, expl_max = self.get_scale_min_max()
            qs = self.rescale_values(qs, q_min, q_max)
            # achieved_values_t = self.rescale_values(achieved_values_t, aim_min, aim_max)
            exploration_values = (exploration_values - expl_min) / (expl_max - expl_min)
        total_val = self.gamma * achieved_values + self.beta * qs + self.sigma * exploration_values
        return total_val, achieved_values, exploration_values, qs
    
    
        

