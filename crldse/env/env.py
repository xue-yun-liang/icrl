import math
from typing import Optional, Union

import numpy as np
import torch

import gym
from gym import logger, spaces
from crldse.env.space import design_space, create_space_crldse
from crldse.env.eval import evaluation_function


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

    FIXME: the value rrange may need to adjust (more modern value)

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


    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination:
    2. Termination:

    ### Arguments

    target: Target platform for accelerator use. e.g. embeded, cloud, pc, workstation .etc
    constraint: Constraint parameters for each dims

    """

    metadata = {}

    def __init__(self, target: str, constraint: dict):
        self.target = target            # the Optimized objects
        self.constraint = constraint    # the constraint list
        self.sample_times = 0

        # init the design space and set constraint parameters
        self.design_space = create_space_crldse()
        assert isinstance(self.design_space, design_space)

        # init some metadate of env
        self.design_space_dimension = self.design_space.get_length()
        self.action_dimension_list = list()
        self.action_limit_list = list()
        for dimension in self.design_space.dimension_box:
            self.action_dimension_list.append(int(dimension.get_scale()))
            self.action_limit_list.append(int(dimension.get_scale() - 1))

        # set eval function
        self.evaluation = evaluation_function(self.target)

    def step(self, step, act):
        # FIXME: fix the code here
        
        # FIXME: next four lines code only for SAC algo
        act = act if torch.is_tensor(act) else torch.as_tensor(act, dtype=torch.float32).view(-1)
        act = torch.softmax(act, dim=-1)
        # sample act based on the probability of the result of act softmax
        act = int(act.multinomial(num_samples=1).data.item())
        
        current_status = self.design_space.get_status()
        next_status = self.design_space.sample_one_dimension(step, act)
        obs = self.design_space.get_obs()

        if step < (self.design_space.get_length() - 1):
            done = False
        else:
            done = True
        
        if not done:
            self.evaluation.update_parameter(next_status, has_memory=True)
            

            # get evaluation info
            runtime= self.evaluation.run_time()*1000
            energy = self.evaluation.energy()

            # coumpute the reward
            if self.goal == "latency":
                reward = 1000 / (runtime * self.constraints.get_punishment())
                result = runtime
            elif self.goal == "energy":
                reward = 1000 / (energy * self.constraints.get_punishment())
                result = energy
            elif self.goal == "latency&energy":
                reward = 1000 / ((runtime * energy) * self.constraints.get_punishment())
                result = runtime * energy

        self.sample_time += 1

        # the step function need to return next_state, reward, done, metadata
        return obs, reward, done, {}

    def reset(self):
        self.design_space.reset_status()
        return self.design_space.get_obs()

    def sample(self, step):
        # here step refer to the idx of the design dimension
        idx = np.random.randint(0, self.design_space.get_dimension_scale(step) - 1)
        # pi if a ont hot tensor, that refer to which dim will be act 
        pi = torch.zeros(int(self.design_space.get_dimension_scale(step)))
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
    dse_env = gym.make("MCPDseEnv-v0", target=target_, constraint=constraint_params)
    dse_env.sample(2)
