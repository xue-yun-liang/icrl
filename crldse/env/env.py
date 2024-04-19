import math
from typing import Optional, Union

import numpy as np
import torch

import gym
from gym import logger, spaces
from crldse.env.space import design_space, create_space_crldse
from crldse.env.eval import evaluation_function
from crldse.config import test_config


class MCPDseEnv(gym.Env):
    """
    ### Description

    This environment corresponds to the concept of multi-core processor design space
    in the exploration of multi-core processor design space. A multi-core processor
    has multiple different components, including the 'core', 'l1i_size', 'l1d_size',
    'l2_size', 'l1d_assoc', 'l1i_assoc', 'l2_assoc', 'sys_clock'.At the beginning,
    each variable is given a parameter, and each parameter can change the performance
    of the entire processor through sampling (this step is called an action).

    ### Action Space

    The action is different for each variable (or dimension). For example, it can be
    changing the core or changing the sys_clock. Due to env having 8 variables,
    the action is a numpy array of (8,), every element is int data type.

    ### Observation Space

    The observation is a `ndarray` with shape `(8,)` with the values corresponding
    to the following 8 variables:

    the value rrange can be adjusted by adjust the dimension_discrete's 'rrange'

    | Num | Observation  | default_value   |  step    |   Min    | Max      |
    |-----|--------------|-----------------|----------|----------|----------|
    | 0   | core         | 1               | 1        | 1        |  16      |
    | 1   | l1i_size     | 2               | i**2     | 2        |  4096    |
    | 2   | l1d_size     | 2               | i**2     | 2        |  4096    |
    | 3   | l2_size      | 64              | i**2     | 64       |  65536   |
    | 4   | l1d_assoc    | 1               | i**2     | 2        |  16      |
    | 5   | l1i_assoc    | 1               | i**2     | 1        |  16      |
    | 6   | l2_assoc     | 1               | i**2     | 1        |  16      |
    | 7   | sys_clock    | 2               | 0.1      | 2.0      |  4.0     |

    ### Rewards
    # the reward 's compute logics:
    reward could be compute by constraints.get_punishment()
    the equations is : R_{s_t} = \prod_{i=1}^{n} O_i(s_t) \prod_{j=1}^{m} (C_j)/(P_j(s_t))^{l_j}
    where the O_i(s_t) is the performance indicator. e.g. the optimization goal "energy" or "latency"
    where the P_j(s_t) is the constraint indicator. 
    where the C_j is the constraint indicators.
    
    **Notes there is a hyperparams threshold_ratio, it as the l_j contorl the total reward

    ### Starting State

    the start state can adjust by dimension_discrete's 'default_value'

    ### Episode End

    The episode ends if any one of the following occurs:

    # FIXME: when the agent will end its acts
    1. Termination:
    2. Termination:

    ### Arguments
    
    Nothing

    """

    metadata = {}

    def __init__(self, config):
        super(MCPDseEnv, self).__init__()
        self.sample_time = 0

        # init the design space and set constraint parameters
        self.design_space = create_space_crldse()
        assert isinstance(self.design_space, design_space)
        self.config = config

        # init some metadate of env
        self.design_space_dimension = self.design_space.get_length()
        self.action_dimension_list = list()
        self.action_limit_list = list()
        for dimension in self.design_space.dimension_box:
            self.action_dimension_list.append(int(dimension.get_scale()))
            self.action_limit_list.append(int(dimension.get_scale() - 1))

        # set eval function
        self.evaluation = evaluation_function(target=self.config.target)
        self.result = 0
        
        self.steps = 0

    def step(self, action):
        print("enter step")
        # FIXME: fix the code here
        
        # FIXME: next four lines code only for SAC algo
        action = action if torch.is_tensor(action) else \
            torch.as_tensor(action, dtype=torch.float32).view(-1)
        action = torch.softmax(action, dim=-1)
        # sample act based on the probability of the result of act softmax
        action = int(action.multinomial(num_samples=1).data.item())
        
        current_status = self.design_space.get_status()
        next_status = self.design_space.sample_one_dimension(self.steps)
        obs = self.design_space.get_obs()
        
        if self.steps < (self.design_space.get_length() - 1):
            done = False
        else:
            done = True
        
        if not done:
            self.evaluation.update_parameter(next_status)
            

            # get evaluation info
            runtime= self.evaluation.run_time()*1000
            energy = self.evaluation.energy()

            # coumpute the reward
            if self.config.goal == "latency":
                reward = self.config.constraints.get_punishment()
                # reward = 1000 / (runtime * self.config.constraints.get_punishment())
                self.result = runtime
            elif self.config.goal == "energy":
                reward = self.config.constraints.get_punishment()
                # reward = 1000 / (energy * self.config.constraints.get_punishment())
                self.result = energy
            elif self.config.goal == "latency&energy":
                reward = self.config.constraints.get_punishment()
                # reward = 1000 / ((runtime * energy) * self.config.constraints.get_punishment())
                self.result = runtime * energy

        self.sample_time += 1
        print(type(obs)," ",type(reward)," ",type(done)," ",type({}))
        # the step function need to return next_state, reward, done, metadata
        return obs, reward, done, {}

    def reset(self):
        self.design_space.reset_status()
        return self.design_space.get_obs()

    def sample(self):
        # here step refer to the idx of the design dimension
        idx = np.random.randint(0, self.design_space.get_dimension_scale(self.steps) - 1)
        # pi if a ont hot tensor, that refer to which dim will be act 
        pi = torch.zeros(int(self.design_space.get_dimension_scale(self.steps)))
        pi[idx] = 1
        return pi


if __name__ == "__main__":
    
    # test code for env wapper
    target_ = "embedded"
    constraint_params = {
        "core": 10,
        "l1i_size": 32,
        "l1d_size": 32,
        "l2_size": 128,
        "l1d_assoc": 2,
        "l1i_assoc": 2,
        "l2_assoc": 2,
        "sys_lock": 3.0,
    }
    env = gym.make("MCPDseEnv-v0", config=test_config())
    obs = env.reset()
    print(obs)
    # for i in range(7):
    # act = env.sample()
    # print('act:{}'.format(act))
    # observation, reward, done, info = env.step(act) # take a random action
    # print('observation:{}, reward:{}, done:{}, info:{}'.format(observation, reward, done, info))
    print(type(env))
    print(env.step([0,1]))
    observation, reward, done, info = env.step([0,0,0,1])
    
