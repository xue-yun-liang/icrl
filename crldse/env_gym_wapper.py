import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


class MCPDseEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
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

    | Num | Observation     | Min                 | Max               |
    |-----|-----------------|---------------------|-------------------|
    | 0   | core            | -4.8                | 4.8               |
    | 1   | l1i_size        | -Inf                | Inf               |
    | 2   | l1d_size        | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | l2_size         | -Inf                | Inf               |
    | 4   | l1d_assoc       | -Inf                | Inf               |
    | 5   | l1i_assoc       | -Inf                | Inf               |
    | 6   | l2_assoc        | -Inf                | Inf               |
    | 7   | sys_clock       | -Inf                | Inf               |

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: 
    2. Termination: 

    ### Arguments
    
    ============ 后续全部加上 ============
    max_cores: Maximum number of cores
    max_l1i_size: max size of l1i size
    min_l2_size: min size of l2 cache 

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)


        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0
        done = terminated
        # the step function need to return next_state, reward, done, metadata
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        return np.array(self.state, dtype=np.float32), {}
    
    def close(self):
        if self.screen is not None:
            # import pygame

            # pygame.display.quit()
            # pygame.quit()
            self.isopen = False