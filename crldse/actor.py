import random
import math

import numpy as np
import torch

from crldse.utils import core


class actor_random():
	def make_policy(self, design_space, dimension_index):
		'''
		return the index of dimension's sample_box 
		rather than the actual value for convinience
		'''
		return random.randint(0,design_space.dimension_box[dimension_index].get_scale()-1)

class actor_e_greedy():
    def __init__(self, design_space, qfunction, ratio) -> None:
        self.design_space = design_space
        self.policy_net = qfunction
        self.greedy_possiblity = 0.7
        self.ratio = ratio
        self.true_best_action_index = 0
        self.true_best_qvalue = 0
        
    def choose_act(self) -> list:
        if(random.random() < self.greedy_possiblity**self.ratio):
        # greedy search best action
        # find the best action in that dimension
        # for act_idx in range(int((self.design_space.get_dimension_scale(dimension_index)))):
            for dim_idx in range(len(self.design_space)):
                state = self.design_space.sample_one_dimension(dim_idx)
                with torch.no_grad():
                    # compute the q value
                    step = (dim_idx+1) / self.design_space.get_length()
                    step = torch.tensor(step).float().view(1)
                    state = core.state_normalize(state, self.design_space)
                    variable = core.state_to_Variable(state)
                    variable = torch.cat((variable, step), dim = -1)

                    qvalue = self.qfunction(variable)
                # compare and find the best q value
                if(qvalue > self.best_qvalue):
                    self.best_action_idx = act_idx
                    self.best_qvalue = qvalue
        else:
            # random choose an action
            # self.best_action_index = random.randint(sample_bottom, sample_top - 1)
            return self.sample_act()
        
    def sample_act(self):
        pi = []
        for i in range(self.design_space.get_length()):
            cur_scale = self.design_space.get_dimension_scale(i)
            sample_idx = np.random.randint(0,  cur_scale - 1)
            pi_i = torch.zeros(int(cur_scale))
            pi_i[sample_idx] = 1
            pi.append(pi_i)
        return pi
        
        

class actor_e_greedy_erdse():
	def __init__(self, design_space):
		self.greedy_possiblity = 0.7
		self.sample_range = int(2)
		self.design_space = design_space
		
	def action_choose(self, qfunction, dimension_index, ratio=1):
		# sample_bottom, sample_top = self.get_sample_range(dimension_index=dimension_index)
		
		if(random.random() < self.greedy_possiblity**ratio):
			# greedy search best action
			self.best_action_index = 0
			self.best_qvalue = 0
			# find the best action in that dimension
			for action_index in range(int((self.design_space.get_dimension_scale(dimension_index)))):
			# for action_index in range(sample_bottom, sample_top):
				state = self.design_space.sample_one_dimension(dimension_index)
				with torch.no_grad():
					# compute the q value
					step = (dimension_index+1) / self.design_space.get_length()
					step = torch.tensor(step).float().view(1)
					state = core.state_normalize(state, self.design_space)
					variable = core.state_to_Variable(state)
					variable = torch.cat((variable, step), dim = -1)

					qvalue = qfunction(variable)
				# compare and find the best q value
				if(qvalue > self.best_qvalue):
					self.best_action_index = action_index
					self.best_qvalue = qvalue
		else:
			# random choose an action
			self.best_action_index = random.randint(0, self.design_space.get_dimension_scale(dimension_index) - 1)
			# self.best_action_index = random.randint(sample_bottom, sample_top - 1)
		return self.best_action_index

	
	def random_action_choose(self, design_space, dimension_index):
		sample_bottom, sample_top = self.get_sample_range(dimension_index=dimension_index)
		# random choose an action
		self.best_action_index = random.randint(0, design_space.get_dimension_scale(dimension_index) - 1)
		# self.best_action_index = random.randint(sample_bottom, sample_top - 1)
		return self.best_action_index

	def best_action_choose(self, qfunction, design_space, dimension_index):
		# greedy search best action
		self.true_best_action_index = 0
		self.true_best_qvalue = 0
		# find the best action in that dimension
		for action_index in range(int((design_space.get_dimension_scale(dimension_index)))):
			state = design_space.sample_one_dimension(dimension_index, action_index)
			with torch.no_grad():
				step = (dimension_index+1) / design_space.get_lenth()
				step = torch.tensor(step).float().view(1)
				state = core.state_normalize(state, design_space)
				variable = core.state_to_Variable(state)
				variable = torch.cat((variable, step), dim = -1)
				
				qvalue = qfunction(variable)
			##### compare and find the best q value
			if(qvalue > self.true_best_qvalue):
				self.true_best_action_index = action_index
				self.true_best_qvalue = qvalue
		return self.true_best_action_index

	def get_sample_range(self, dimension_index):
     	# python range is [a,b), so the up bound requre a +1
		current_index = self.design_space.get_dimension_current_index(dimension_index)
		sample_bottom = max(0, current_index - self.sample_range)
		sample_top = min(int(self.design_space.get_dimension_scale(dimension_index)), current_index + self.sample_range + 1)		
		return sample_bottom, sample_top
		
    

class actor_policyfunction():
    def __init__(self, policyfunction, design_space) -> None:

        self.policy_net = policyfunction
        self.design_space = design_space
    
    def action_choose(self, policyfunction, design_space, dimension_index):
        state = design_space.get_state()
        state_normalization = core.state_normalize(state, design_space)
        probs = policyfunction(core.state_to_Variable(state_normalization), dimension_index)
        use_noise = False
        if(use_noise):		
            noise = torch.normal(mean = torch.zeros_like(probs), std = 0.005)
            probs_noise = probs + noise
            probs_noise = torch.clamp(probs_noise, 0, 1)
            action_index = probs_noise.multinomial(num_samples = 1).data
        else:
            action_index = probs.multinomial(num_samples = 1).data
        # compute entropy
        entropy = -(probs * probs.log()).sum()
        # use multinomial to realize the sampling of policy function
        action_index_tensor = core.index_to_one_hot(len(probs), action_index)
        # use onehot index to restore compute graph
        prob_sampled = (probs * action_index_tensor).sum()
        log_prob_sampled = prob_sampled.log()

        return entropy, action_index, log_prob_sampled


    def action_choose_with_no_grad(self, policyfunction, design_space, dimension_index, std = 0.1, is_train = True):
        state = design_space.get_state()
        with torch.no_grad():
            state_normalization = core.state_normalize(state, design_space)
            probs = policyfunction(core.state_to_Variable(state_normalization), dimension_index)

            if(is_train):
                model = {"name":"normal", "param":0.1}
                if(model["name"] == "normal"):
                    noise = torch.normal(mean = torch.zeros_like(probs), std = std)
                    probs_noise = probs + noise
                    probs_noise = torch.clamp(probs_noise, 0, 1)
                    # probs_noise = abs(probs_noise)
                    # probs_noise = probs_noise/probs_noise.sum()
                    # probs_noise = probs
                elif(model["name"] == "one_hot"):
                    noise = torch.normal(mean = torch.zeros_like(probs), std = model["param"])
                    probs_noise = probs + noise
                    probs_noise = torch.clamp(probs_noise, 0, 1)
                    # probs_noise = abs(probs_noise)
                    # probs_noise = probs_noise/probs_noise.sum()
            else:
                probs_noise = probs

            # probs_noise = torch.abs(probs + noise)
            # probs_noise = torch.nn.functional.softmax(probs + noise)
            # use multinomial to realize the sampling of policy function
            action_index = probs_noise.multinomial(num_samples = 1).data

        return action_index, probs_noise[action_index].data


    def action_choose_DDPG(self, policyfunction, design_space, dimension_index):
        with torch.no_grad():
            state = design_space.get_state()
            state = core.state_normalize(state, design_space)
            probs = policyfunction(core.state_to_Variable(state), dimension_index)
            probs_softmax = torch.softmax(probs, dim = -1)
            # use multinomial to realize the sampling of policy function
            action_index = probs_softmax.multinomial(num_samples = 1).data
            # action_index = torch.argmax(probs)
        return action_index, probs

if __name__=='__main__':
    from crldse.env.space import create_space
    from crldse.utils.core import read_config
    from crldse.net import mlp_policyfunction
    conf_data = read_config('./env/config.yaml')
    dse_space = create_space(conf_data)
    action_scale_list = list()
    for dimension in dse_space.dimension_box:
        action_scale_list.append(int(dimension.get_scale()))
    policyfunction = mlp_policyfunction(dse_space.get_length(), action_scale_list)
    ac = actor_policyfunction()
    action, probs_noise = ac.action_choose_with_no_grad(policyfunction, dse_space, 0)
    print(action)
    print(probs_noise)