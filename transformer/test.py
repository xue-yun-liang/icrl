import torch
import random
import numpy as np
import datetime
from transformer.model import PositionalEncoding, Transformer


# obs           dict, metrics(gem5 to mcapt output) & action from the *previous* timestep
# action        int , actor.action_choose() return a new action index
# prev_action   np.ndarray , action from the *previous* timestep
# reward        float, a float number compute by reward function
# reset         bool , "soft resets" (only used in RL^2 inputs)
# goal          string, application platform -> embedded, PC, LOT, and server.etc
# time          string, the time of the current training session

obs = {
    "core": "3",
    "benchmarksize": "",
    "l1i_size": "256",
    "l1d_size": "256",
    "l2_size": "64",
    "l1d_assoc": "8",
    "l1i_assoc": "8",
    "l2_assoc": "8",
    "sys_clock": "2",
}
action = 1  # index in sample box
prev_action = np.array([random.randint(0, 100) for _ in range(10)])
reward = random.randint(0, 100)
reset = False
goal = "embedded"
current_time = datetime.datetime.now()


def concatenate_input(obs, action, prev_action, reward, reset, goal, current_time):
    obs_str = "".join([str(value) for value in obs.values()])
    action_str = str(action)
    prev_action_str = ",".join(map(str, prev_action))
    reward_str = str(reward)
    reset_str = str(reset)
    goal_str = str(goal)
    current_time_str = str(current_time)

    concatenated_str = (
        obs_str
        + action_str
        + prev_action_str
        + reward_str
        + reset_str
        + goal_str
        + current_time_str
    )

    return concatenated_str


input = concatenate_input(obs, action, prev_action, reward, reset, goal, current_time)

logger.info(input)

pos_embed = PositionalEncoding(input)
new_encode = Transformer()
encoder_output = new_encode.forward(pos_embed)
logger.info(new_encode)
