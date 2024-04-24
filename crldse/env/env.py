import math
from typing import Optional, Union

import numpy as np
import torch
import gym

from crldse.env.space import design_space, create_space
from crldse.env.eval import evaluation_function
from crldse.env.constraints import create_constraints_conf
from crldse.utils.core import read_config, sample_index_from_2d_array


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

    def __init__(self):
        super(MCPDseEnv, self).__init__()
        self.sample_times = 0

        # init the design space and set constraint parameters
        self.config = read_config('./config.yaml')
        self.constraints_conf = create_constraints_conf(config_data=self.config)
        self.design_space = create_space(config_data=self.config)
        assert isinstance(self.design_space, design_space)
        
        # init some metadate of env
        self.design_space_dimension = self.design_space.get_length()
        self.action_dimension_list = list()
        self.action_limit_list = list()
        for dimension in self.design_space.dimension_box:
            self.action_dimension_list.append(int(dimension.get_scale()))
            self.action_limit_list.append(int(dimension.get_scale() - 1))

        # set eval function
        self.eval_func = evaluation_function()

    def step(self, action):                       
        # step1: give the 'act' for design_space, which return the next_obs
        sample_idx = sample_index_from_2d_array(act)
        self.design_space.set_state(sample_idx)
        next_obs = self.design_space.get_obs()
        
        done = False if self.sample_times <= 50 else True
        if not done:
            # step2: the eval funct give the performance result for next obs
            eval_res = self.eval_func.eval(next_obs)
            # eval_res has {'latency', 'Area', 'energy', 'power'}
            latency = eval_res['latency']*1000
            energy = eval_res['energy']
            
            # TODO: step3: update the config.constraints's performance values 
            self.constraints_conf.constraints.update(eval_res)
            # coumpute the reward
            reward = self.constraints_conf.constraints.get_punishment()
            punishment = 1 if reward == 0 else reward
            reward = 1000 / (latency * punishment)
            if self.config['goal'] == "latency":
                self.result = latency
            elif self.config['goal'] == "energy":
                self.result = energy
            elif self.config['goal'] == "latency&energy":
                self.result = latency * energy

        self.sample_times += 1
        
        # finally, the step func return next_state, reward, done, metadata
        return next_obs, reward, done, {}

    def reset(self):
        self.design_space.reset_status()
        return self.design_space.get_obs()

    def sample_act(self):
        pi = []
        for i in range(self.design_space.get_length()):
            cur_scale = self.design_space.get_dimension_scale(i)
            sample_idx = np.random.randint(0,  cur_scale - 1)
            pi_i = torch.zeros(int(cur_scale))
            pi_i[sample_idx] = 1
            pi.append(pi_i)
        return pi


if __name__ == "__main__":
    # test code for env wapper
    env = gym.make("MCPDseEnv-v0")
    obs = env.reset()
    print(obs)
    for i in range(10):
        print("the {}th epoch".format(i))
        act = env.sample()
        print(act)
        observation, reward, done, info = env.step(act) # take a random action
        print('observation:{}, reward:{}, done:{}, info:{}'.format(observation, reward, done, info))
    
