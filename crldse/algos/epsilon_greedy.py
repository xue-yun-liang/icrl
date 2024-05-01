import torch
import numpy as np
import gym
from crldse.net import mlp_policyfunction
from crldse.utils.core import get_log_prob
# import gym_wapper
class epsilon_greedy():
    def __init__(self, ratio, epsilon) -> None:
        """
        implement an algos it act int sample process during using other rl algos
        """
        self.ratio = ratio
        self.epsilon = epsilon
        self.env = gym.make("gym_wapper/MCPDseEnv-v0")
        _ = self.env.reset()
        action_scale_list = list()
        for dimension in self.env.design_space.dimension_box:
            action_scale_list.append(int(dimension.get_scale()))
        self.policyfunction = mlp_policyfunction(self.env.design_space.get_length(), action_scale_list)
        self.opt = torch.optim.Adam(self.policyfunction.parameters(),lr=0.03)
        
    def choose_act(self):
        # FIXME: assume the output size is same
        if self.epsilon > np.random.random():
            return self.design_space.sample_state()
        else:
            best_act = mlp_policyfunction()
    
    def get_reward(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        return reward
    
    def update_policy_net(self):
        # TODO: add logits for loss of compute
        loss = torch.tensor(0)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
if __name__ == '__main__':
    eps_greed = epsilon_greedy(ratio=0.98, epsilon=0.05)
    
        
        