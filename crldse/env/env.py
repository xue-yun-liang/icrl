import math
from typing import Optional, Union

import numpy as np
import torch
import gym
import yaml

from crldse.env.space import design_space, create_space
from crldse.env.eval import evaluation_function
from crldse.env.constraints import create_constraints_conf
from crldse.utils.core import read_config


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
        
        self.result = 0
        self.steps = 6

    def step(self, action):
        print("enter step")
        # FIXME: fix the code here
        # the logics in  here (ignore all processes for data):
        # step1: give the action for design space, then design space give a new obs for next state
        # step2: the evaluation function give the performance result for next state
        # step3: the config.constraints update the performance values by the new performance result \
            # and the reward could be compute by the config.constraints.get_punishment()
                                    
        # action(policy's reult) ---(step1:transition)--->  next_state (sampled in design space) ---------------------------------------|
        # reward <---(step3: update the config.constraints)--- evalued result(preformace's indicators)<---(step2:evaluation_function)---| 
        
        # NOTE* when we got the next state, need to adjust whether we arrive the 'done' state
        
        # FIXME: next four lines code only for SAC algo
        
        # For pre-processing of actions
        action = action if torch.is_tensor(action) else \
            torch.as_tensor(action, dtype=torch.float32).view(-1)
        action = torch.softmax(action, dim=-1)
        # sample actual current act based on the probability of the result of act softmax
        action = int(action.multinomial(num_samples=1).data.item())
        
        # step1: give the action for design space, get the next_statue
        current_status = self.design_space.get_status()
        # FIXME: Is it reasonable to set the logic of state transition to random sampling ?
        # Until here, the action is equal to a tensor[idx_dim1, idx_dim2, idx_dim3, ..., idx_dimn]
        # where the n refer to the design dims. Under current's occusion n = 8.
        
        next_obs = self.design_space.sample_one_dimension(self.steps)
        obs = self.design_space.get_obs()
        
        # step1.5:
        if self.steps < (self.design_space.get_length() - 1):
            done = False
        else:
            done = True
        
        if not done:
            eval_res = self.eval_func.eval(next_obs)
            
            # step2: get the performance result by evaluation_function

            # get evaluation info
            runtime= self.evaluation.run_time()*1000
            energy = self.evaluation.energy()
            
            # TODO: step3: update the config.constraints's performance values 
            self.constraints_conf.constraints.update(eval_res)
            # coumpute the reward
            reward = self.constraints_conf.constraints.get_punishment()
            punishment = 1 if reward == 0 else reward
            reward = 1000 / (runtime * punishment)
            if self.config['goal'] == "latency":
                self.result = runtime
            elif self.config['goal'] == "energy":
                self.result = energy
            elif self.config['goal'] == "latency&energy":
                self.result = runtime * energy

        self.sample_times += 1
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
    env = gym.make("MCPDseEnv-v0")
    obs = env.reset()
    print(obs)
    for i in range(10):
        print("the {}th epoch".format(i))
        act = env.sample()
        print(act)
        observation, reward, done, info = env.step(act) # take a random action
        print('observation:{}, reward:{}, done:{}, info:{}'.format(observation, reward, done, info))
    
