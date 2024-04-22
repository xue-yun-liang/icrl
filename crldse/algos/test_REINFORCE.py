import random
import pdb
import copy
from multiprocessing import Pool

import gym
import numpy as np
import torch
import xlwt
import yaml

from crldse.env.space import dimension_discrete, design_space, create_space
from crldse.env.eval import evaluation_function
from crldse.actor import actor_e_greedy, actor_policyfunction
from crldse.net import mlp_policyfunction
from crldse.env.constraints import create_constraints_conf, print_config
from crldse.env.eval import evaluation_function
from crldse.utils import core


debug = False


class REINFORCEBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        self.adv_buf = torch.as_tensor(self.adv_buf, dtype=torch.float32)
        self.adv_buf = (self.adv_buf - torch.mean(self.adv_buf)) / torch.std(self.adv_buf)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class RLDSE:
    def __init__(self, iindex):

        # 1. Logger setup
        self.iindex = iindex

        # 2. Random seed setting
        seed = self.iindex * 10000
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # assign goal, platgorm and constraints
        with open('config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)

        self.constraints_conf = create_constraints_conf(config_data)
        print_config(self.constraints_conf)
        
        # record the train process
        # self.workbook = xlwt.Workbook(encoding="ascii")
        # self.worksheet = self.workbook.add_sheet("1")
        # self.worksheet.write(0, 0, "period")
        # self.worksheet.write(0, 1, "return")
        # self.worksheet.write(0, 2, "loss")

        ## 3. Environment instantiation
        self.env = gym.make('MCPDseEnv-v0')

        # define the hyperparameters
        self.SAMPLE_PERIOD_BOUND = 1
        self.GEMA = 0.999  # RL parameter, discount ratio
        self.ALPHA = 0.001  # RL parameter, learning step rate
        self.THRESHOLD_RATIO = 2  # 0.05
        self.BATCH_SIZE = 1
        self.BASE_LINE = 0
        self.ENTROPY_RATIO = 0
        self.PERIOD_BOUND = 500

        # 4. constructing the ac pytoch module
        # initial mlp_policyfunction, every action dimension owns a policyfunction
        action_scale_list = list()
        for dimension in self.DSE_action_space.dimension_box:
            action_scale_list.append(int(dimension.get_scale()))
        self.policyfunction = mlp_policyfunction(self.DSE_action_space.get_length(), action_scale_list)

        ##initial e_greedy_policy_function
        self.actor = actor_policyfunction()

        ##initial evaluation
        default_state = {
            "core": 3,
            "l1i_size": 256,
            "l1d_size": 256,
            "l2_size": 64,
            "l1d_assoc": 8,
            "l1i_assoc": 8,
            "l2_assoc": 8,
            "sys_clock": 2,
        }
        self.evaluation = evaluation_function(default_state, '../env/sim_config.yaml')

        # 7. Making pytorch optimizers
        self.policy_optimizer = torch.optim.Adam(self.policyfunction.parameters(),lr=self.ALPHA)

        #### loss replay buffer, in order to record and reuse high return trace
        self.loss_buffer = list()

        # 5. Instantiating the experience buffer(for compute the loss)
        #### data vision related
        self.objectvalue_list = list()
        self.objectvalue_list.append(0)
        self.power_list = list()
        self.period_list = list()
        self.period_list.append(-1)
        self.best_objectvalue = 10000
        self.best_objectvalue_list = list()
        self.best_objectvalue_list.append(self.best_objectvalue)
        self.all_objectvalue = list()
        self.all_objectvalue2 = list()
        self.best_objectvalue2 = 10000
        self.best_objectvalue2_list = list()
        self.best_objectvalue2_list.append(self.best_objectvalue)

        self.action_array = list()
        self.reward_array = list()

        # context encoder module
        # the encoder's input = obs + action + reward + reset + goal + time
        # a exmaple:
        # obs           dict, metrics(gem5 to mcapt output) & action from the *previous* timestep
        # action        int , actor.action_choose() return a new action index
        # prev_action   np.ndarray , action from the *previous* timestep
        # reward        float, a float number compute by reward function
        # reset         bool , "soft resets" (only used in RL^2 inputs)
        # goal          string, application platform -> embedded, PC, LOT, and server.etc
        # time          string, the time of the current training session
        
        # give the all info for encoder module
        # the decoder will change the weight of info during training
        # FIXME: think about how to embedded the transformer into training process
        # self.embedding = model.PositionalEncoding()
        # self.meta_encoder = model.Transformer()
        # self.meta_encoder_lr = 0.05

    # 6. Setting up callble loss functions that also provide disgnostics specfic to algo
    def train(self):
        current_status = dict()  # S
        next_status = dict()  # S'

        loss = torch.tensor(0)  # define loss function
        batch_idx = 0

        period_bound = self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND
        for period in range(self.PERIOD_BOUND):
            print(f"period:{period}", end="\r")
            self.DSE_action_space.reset_status()

            # store log_prob, reward and return
            entropy_list = list()
            log_prob_list = list()
            reward_list = list()
            return_list = list()
            entropy_loss_list = list()

            for step in range(self.DSE_action_space.get_lenth()):
                # get status from S
                current_status = self.DSE_action_space.get_status()

                # use policy function to choose action and acquire log_prob
                entropy, action, log_prob_sampled = self.actor.action_choose(
                    self.policyfunction, self.DSE_action_space, step
                )
                entropy_list.append(entropy)
                log_prob_list.append(log_prob_sampled)

                # take action and get next state S'
                next_status = self.DSE_action_space.sample_one_dimension(step, action)

                #### in MC method, we can only sample in last step
                # and compute reward R

                if step < (
                    self.DSE_action_space.get_lenth() - 1
                ):  # delay reward, only in last step the reward will be asigned
                    reward = float(0)
                    reward2 = float(0)
                else:
                    
                    metrics = self.evaluation.eval(next_status)
                    if metrics != None:

                        energy = metrics["latency"]
                        area = metrics["Area"]
                        runtime = metrics["latency"]
                        power = metrics["power"]
                        self.constraints.update({"AREA": area})

                        reward = 1000 / (runtime * self.constraints.get_punishment())
                        objectvalue = runtime
                        objectvalue2 = power
                    else:
                        reward = 0

                    #### recording
                    if (
                        objectvalue < self.best_objectvalue
                        and self.constraints.is_all_meet()
                    ):
                        self.best_objectvalue = objectvalue
                        print(f"best_status:{objectvalue}")
                    if self.constraints.is_all_meet():
                        self.all_objectvalue.append(objectvalue)
                        self.all_objectvalue2.append(objectvalue2)
                    else:
                        self.all_objectvalue.append(10000)
                        self.all_objectvalue2.append(10000)
                    self.best_objectvalue_list.append(self.best_objectvalue)
                    self.period_list.append(period)
                    self.objectvalue_list.append(reward)
                    self.power_list.append(power)

                reward_list.append(reward)

                # assign next_status to current_status
                current_status = next_status

            self.action_array.append(self.DSE_action_space.get_action_list())
            self.reward_array.append(reward)

            # compute and record return
            return_g = 0
            T = len(reward_list)
            for t in range(T):
                return_g = reward_list[T - 1 - t] + self.GEMA * return_g
                return_list.append(torch.tensor(return_g).reshape(1))
            return_list.reverse()
            # self.worksheet.write(period + 1, 0, period)
            # self.worksheet.write(period + 1, 1, return_list[0].item())

            # compute and record entropy_loss
            entropy_loss = torch.tensor(0)
            T = len(return_list)
            for t in range(T):
                retrun_item = -1 * log_prob_list[t] * (return_list[t] - self.BASE_LINE)
                entropy_item = -1 * self.ENTROPY_RATIO * entropy_list[t]
                entropy_loss = entropy_loss + retrun_item + entropy_item
            entropy_loss = entropy_loss / T

            loss = loss + entropy_loss
            batch_index = batch_index + 1

            # step update policyfunction
            if period % self.BATCH_SIZE == 0:
                loss = loss / self.BATCH_SIZE
                # logger.info(f"entropy_loss:{entropy_loss}")
                self.worksheet.write(int(period / self.BATCH_SIZE) + 1, 2, loss.item())
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()

                loss = torch.tensor(0)
                batch_index = 0
        # end for-period
        self.workbook.save("record/new_reward&return/REINFORCE_reward_record.xls")

    # end def-train

    def test(self):
        pass

# 10. running the main loop of the algorithms
def run(iindex):
    print(f"---------------TEST{iindex} START---------------")
    DSE = RLDSE(iindex)
    DSE.train()
    print(f"---------------TEST{iindex} END---------------")


if __name__ == "__main__":
    TEST_BOUND = 4
    for iindex in range(3, TEST_BOUND):
        run(iindex)
