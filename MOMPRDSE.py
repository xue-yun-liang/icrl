from space import dimension_discrete
from space import design_space
from space import create_space_gem5
from space import tsne3D
from actor import actor_policyfunction, get_log_prob
from evaluation import evaluation_function
from config import my_test_config
from multiprocessing import Pool

from mlp import mlp_fillter
from gem5_mcpat_evaluation import evaluation
import torch
import random
import numpy
import numpy as np
import pdb
import copy
import xlwt

debug = False

weight_power = 0.9
weight_runtime = 0.1


class mlp_policyfunction(torch.nn.Module):
    def __init__(self, space_lenth, action_scale_list):
        super(mlp_policyfunction, self).__init__()
        self.space_lenth = space_lenth
        self.action_scale_list = action_scale_list
        self.fc1 = torch.nn.Linear(self.space_lenth + 1, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        # layer fc3 is a multi-output mlp
        self.fc3 = list()
        for action_scale in self.action_scale_list:
            self.fc3.append(torch.nn.Linear(64, action_scale))
        self.fc3 = torch.nn.ModuleList(self.fc3)

    def forward(self, qfunction_input, dimension_index):
        input = torch.cat(
            (qfunction_input, torch.tensor(dimension_index).float().view(1)), dim=-1
        )
        out1 = torch.nn.functional.relu(self.fc1(input))
        out2 = torch.nn.functional.relu(self.fc2(out1))
        out3 = self.fc3[dimension_index](out2)
        return torch.nn.functional.softmax(out3, dim=-1)


class RLDSE:
    def __init__(self, iindex):

        self.iindex = iindex

        seed = self.iindex * 10000
        atype = 4

        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

        #### step1 assign model
        self.config = my_test_config()
        # self.nnmodel = self.config.nnmodel
        # self.layer_num = self.config.layer_num
        self.has_memory = True

        #### step2 assign platform
        # constrain
        # self.target = self.config.target
        self.constraints = self.config.constraints

        #### step3 assign goal
        # self.goal = self.config.goal
        self.config.config_check()

        self.workbook = xlwt.Workbook(encoding="ascii")
        self.worksheet = self.workbook.add_sheet("1")
        self.worksheet.write(0, 0, "period")
        self.worksheet.write(0, 1, "return")
        self.worksheet.write(0, 2, "loss")

        ## initial DSE_action_space
        self.DSE_action_space = create_space_gem5()

        # define the hyperparameters
        self.PERIOD_BOUND = 500
        self.SAMPLE_PERIOD_BOUND = 4
        self.GEMA = 0.999  # RL parameter, discount ratio
        self.ALPHA = 0.001  # RL parameter, learning step rate
        self.THRESHOLD_RATIO = 2
        self.BATCH_SIZE = 1
        self.BASE_LINE = 0
        self.ENTROPY_RATIO = 0.1

        self.fillter_train_interval = 500

        self.stack_record = 0

        # initial mlp_policyfunction, every action dimension owns a policyfunction
        # TODO:share the weight of first two layer among policyfunction
        action_scale_list = list()
        for dimension in self.DSE_action_space.dimension_box:
            action_scale_list.append(int(dimension.get_scale()))

        # ++++++++++++++++++++++first weightpolicy++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.policyfunction = mlp_policyfunction(
            self.DSE_action_space.get_lenth(), action_scale_list
        )
        self.policyfunction_2 = mlp_policyfunction(
            self.DSE_action_space.get_lenth(), action_scale_list
        )
        self.policyfunction_3 = mlp_policyfunction(
            self.DSE_action_space.get_lenth(), action_scale_list
        )

        ##initial e_greedy_policy_function
        self.actor = actor_policyfunction()

        ##initial evaluation
        # self.evaluation = evaluation_function(self.nnmodel, self.target)

        ##initial optimizer

        # +++++++++++++++++++++++++++++++first weight+++++++++++++++++++++++++++++++++++
        self.policy_optimizer = torch.optim.Adam(
            self.policyfunction.parameters(),
            lr=self.ALPHA,
        )
        self.policy_optimizer_2 = torch.optim.Adam(
            self.policyfunction_2.parameters(),
            lr=self.ALPHA,
        )
        self.policy_optimizer_3 = torch.optim.Adam(
            self.policyfunction_3.parameters(),
            lr=self.ALPHA,
        )

        self.replay_buffer = list()
        self.replay_buffer2 = list()
        self.replay_buffer3 = list()
        #### replay buffer, in order to record and reuse high return trace
        #### buffer is consist of trace list
        self.return_buffer = list()
        self.status_buffer = list()
        self.action_buffer = list()
        self.step_buffer = list()
        self.probs_noise_buffer = list()
        # +++++++++++++++++++++++++++++++++++++++++second++++++++++++++++++++++++++++++++++
        self.return2_buffer = list()
        self.status2_buffer = list()
        self.action2_buffer = list()
        self.step2_buffer = list()
        self.probs_noise2_buffer = list()
        # +++++++++++++++++++++++++++++++++++third+++++++++++++++++++++++++++++++++++++++
        self.return3_buffer = list()
        self.status3_buffer = list()
        self.action3_buffer = list()
        self.step3_buffer = list()
        self.probs_noise3_buffer = list()

        #### data vision related
        self.objectvalue_list = list()
        self.objectvalue_list.append(0)
        self.objectvalue_list2 = list()
        self.objectvalue_list2.append(0)

        self.objectvalue2_list = list()
        self.objectvalue2_list.append(0)

        self.period_list = list()
        self.period_list.append(-1)
        self.best_objectvalue = 10000
        self.best_objectvalue_list = list()
        self.best_objectvalue_list.append(self.best_objectvalue)
        self.all_objectvalue = list()
        self.all_objectvalue2 = list()
        self.best_objectvalue2 = 10000
        self.best_objectvalue2_list = list()
        self.best_objectvalue2_list.append(self.best_objectvalue2)

        self.power_list = list()

        self.action_array = list()
        self.reward_array = list()
        self.reward2_array = list()
        self.reward3_array = list()
        # ++++++++++++++++++++++++++fillter+++++++++++++++++++++++++++++
        self.fillter = mlp_fillter(self.DSE_action_space.get_lenth())
        #### fillter buffer
        self.fillter_obs_buffer = list()
        self.fillter_reward_buffer = list()

        self.fillter2 = mlp_fillter(self.DSE_action_space.get_lenth())
        #### fillter buffer
        self.fillter_obs_buffer2 = list()
        self.fillter_reward_buffer2 = list()

        self.fillter3 = mlp_fillter(self.DSE_action_space.get_lenth())
        #### fillter buffer
        self.fillter_obs_buffer3 = list()
        self.fillter_reward_buffer3 = list()
        self.fillter_list = list()

    def train_fillter(self, fillter, obs_list, reward_list):
        print(
            f"**************  Training the fillter, now we have {len(obs_list)} samples   ******************"
        )
        data_size = len(obs_list)
        batch_size = 50
        epoch_time = 1000

        refuse_record = 0
        reward_list_back = copy.deepcopy(reward_list)
        target_list = list()

        reward_list_back.sort()
        self.best_reward = max(reward_list)
        self.low_reward = reward_list_back[int(0.7 * data_size)]
        print(
            f"***************   refuse_reward = {self.low_reward}, best_reward = {self.best_reward}     ******************"
        )
        for reward in reward_list:
            if reward <= self.low_reward:
                target_list.append(0)
                refuse_record += 1
            else:
                target_list.append(1)
        # print(f"***************   refuse_ratio = {refuse_record/data_size}     ******************")

        temp_optimizer = torch.optim.Adam(fillter.parameters(), 0.001)
        loss_function = torch.nn.CrossEntropyLoss()
        data_all = np.array(obs_list)
        target_all = np.array(target_list)

        fillter.train()
        for epoch in range(epoch_time):
            idxs = np.random.randint(0, int(0.8 * data_size), size=batch_size)
            data = torch.as_tensor(data_all[idxs], dtype=torch.float32)
            target = torch.as_tensor(target_all[idxs], dtype=torch.long)

            predict = fillter(data)
            loss = loss_function(predict, target)
            if epoch % 200 == 0:
                print(f"loss:{loss}")

            temp_optimizer.zero_grad()
            loss.backward()
            temp_optimizer.step()
        ##test

        fillter.eval()
        test_range = list(range(int(0.8 * data_size), data_size))
        data = torch.as_tensor(data_all[test_range], dtype=torch.float32)
        reward = np.array(reward_list)[test_range]
        predict = fillter(data)

        t_predict = list()
        for i in predict:
            if i[0] > 0.9:
                t_predict.append(0)
            else:
                t_predict.append(1)
        t_target = target_all[test_range]
        error = 0
        for i, j, d, r in zip(t_predict, t_target, data, reward):
            if i != j:
                error += 1
        print(
            f"************      accucy = {1 - error / len(test_range)}    ***************"
        )

    # pdb.set_trace()

    def train(self):
        stack_renew = 0
        current_status = dict()  # S
        next_status = dict()  # S'
        reward_log_name = (
            "record/objectvalue_double_three_policy/"
            + "reward_"
            + str(self.iindex)
            + ".txt"
        )
        reward_log = open(reward_log_name, "w")
        period = 0
        period_bound = self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND
        while period < self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND:
            print(period)
            print(f"period:{period}", end="\r")
            # here may need a initial function for action_space
            self.DSE_action_space.status_reset()

            # store status, log_prob, reward and return
            status_list = list()
            step_list = list()
            action_list = list()

            reward_list = list()
            reward2_list = list()
            reward3_list = list()

            return_list = list()
            return2_list = list()
            return3_list = list()

            probs_noise_list = list()

            for step in range(self.DSE_action_space.get_lenth()):
                step_list.append(step)
                # get status from S
                current_status = self.DSE_action_space.get_status()
                status_list.append(current_status)
                change_period = period % 3
                if change_period < 1:
                    signol = 1
                elif change_period >= 1 and change_period < 2:
                    signol = 2
                else:
                    signol = 3

                # use policy function to choose action and acquire log_prob
                action, probs_noise = self.actor.action_choose_with_no_grad_3(
                    self.policyfunction,
                    self.policyfunction_2,
                    self.policyfunction_3,
                    self.DSE_action_space,
                    step,
                    signol,
                )
                action_list.append(action)
                probs_noise_list.append(probs_noise)

                # take action and get next state S'
                next_status = self.DSE_action_space.sample_one_dimension(step, action)

                #### in MC method, we can only sample in last step
                #### and compute reward R

                # TODO:design a good reward function
                if step < (
                    self.DSE_action_space.get_lenth() - 1
                ):  # delay reward, only in last step the reward will be asigned
                    reward = float(0)
                    reward2 = float(0)
                    reward3 = float(0)
                    reward_list.append(reward)
                    reward2_list.append(reward2)
                    reward3_list.append(reward3)
            print(f"policy:{signol}")
            self.fillter_train_flag = (
                (period - self.SAMPLE_PERIOD_BOUND) % 50 == 0
            ) and (period - self.SAMPLE_PERIOD_BOUND != 0)

            if period >= self.SAMPLE_PERIOD_BOUND and self.fillter_train_flag:
                print(f"**************  Training the fillter 1    ******************")
                self.train_fillter(
                    self.fillter, self.fillter_obs_buffer, self.fillter_reward_buffer
                )
            obs = self.DSE_action_space.get_obs()
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            self.fillter.eval()
            predict = self.fillter(t_obs)

            if period >= self.SAMPLE_PERIOD_BOUND and self.fillter_train_flag:
                print(f"**************  Training the fillter 2    ******************")
                self.train_fillter(
                    self.fillter2, self.fillter_obs_buffer2, self.fillter_reward_buffer2
                )
            obs = self.DSE_action_space.get_obs()
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            self.fillter2.eval()
            predict2 = self.fillter2(t_obs)

            if period >= self.SAMPLE_PERIOD_BOUND and self.fillter_train_flag:
                print(f"**************  Training the fillter 3    ******************")
                self.train_fillter(
                    self.fillter3, self.fillter_obs_buffer3, self.fillter_reward_buffer3
                )
            obs = self.DSE_action_space.get_obs()
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            self.fillter3.eval()
            predict3 = self.fillter3(t_obs)

            if signol == 1:
                if (predict[0][0] > 0.9) and not self.fillter_train_flag:
                    self.stack_record += 1
                    if self.stack_record < 50:
                        # print(f"stack_record:{self.stack_record}")
                        stack_renew = stack_renew + 1
                        continue
                    else:
                        self.stack_record = 0
                        pass
                else:
                    self.stack_record = 0

            elif signol == 2:
                if (predict2[0][0] > 0.9) and not self.fillter_train_flag:
                    self.stack_record += 1
                    if self.stack_record < 50:
                        # print(f"stack_record:{self.stack_record}")
                        stack_renew = stack_renew + 1
                        continue
                    else:
                        self.stack_record = 0
                        pass
                else:
                    self.stack_record = 0

            else:
                if (predict3[0][0] > 0.9) and not self.fillter_train_flag:
                    self.stack_record += 1
                    if self.stack_record < 50:
                        # print(f"stack_record:{self.stack_record}")
                        stack_renew = stack_renew + 1
                        continue
                    else:
                        self.stack_record = 0
                        pass
                else:
                    self.stack_record = 0

            if next_status in self.fillter_list:
                print("already in fillter_list")
                continue
            else:
                self.fillter_list.append(next_status)

            metrics = evaluation(next_status)
            if metrics != None:

                energy = metrics["latency"]
                area = metrics["Area"]
                runtime = metrics["latency"]
                power = metrics["power"]
                self.constraints.update({"AREA": area})

                reward_runtime = 6.25 / (
                    runtime / 0.004028 * self.constraints.get_punishment()
                )
                reward_power = 4 / (
                    power / 44.257917 * self.constraints.get_punishment()
                )
                objectvalue = runtime
                objectvalue2 = power

            else:
                reward_runtime = 0
                reward_power = 0
            print(reward_runtime, reward_power)

            reward = reward_runtime

            reward2 = (reward_power * reward_runtime) ** 0.5

            reward3 = reward_power

            # print(reward,reward3)

            #### recording
            if period < self.SAMPLE_PERIOD_BOUND:
                pass
            else:
                if (
                    objectvalue < self.best_objectvalue
                    and self.constraints.is_all_meet()
                ):
                    self.best_objectvalue = objectvalue
                if (
                    objectvalue2 < self.best_objectvalue2
                    and self.constraints.is_all_meet()
                ):
                    self.best_objectvalue2 = objectvalue2

                    # print(f"best_status:{objectvalue,objectvalue2 ,power, DSP, BW, BRAM}")
                if self.constraints.is_all_meet():
                    self.all_objectvalue.append(objectvalue)
                    self.all_objectvalue2.append(objectvalue2)
                else:
                    self.all_objectvalue.append(10000)
                    self.all_objectvalue2.append(10000)

                self.best_objectvalue_list.append(self.best_objectvalue)
                self.best_objectvalue2_list.append(self.best_objectvalue2)
                self.period_list.append(period - self.SAMPLE_PERIOD_BOUND)
                self.objectvalue_list.append(reward)
                self.objectvalue_list2.append(reward2)
                self.power_list.append(power)
                print(f"{period}\t{reward}", end="\n", file=reward_log)

            reward_list.append(reward)
            reward2_list.append(reward2)
            reward3_list.append(reward3)

            obs = self.DSE_action_space.get_obs()
            self.fillter_obs_buffer.append(obs)
            self.fillter_reward_buffer.append(reward)

            self.fillter_obs_buffer2.append(obs)
            self.fillter_reward_buffer2.append(reward2)

            self.fillter_obs_buffer3.append(obs)
            self.fillter_reward_buffer3.append(reward3)

            self.action_array.append(self.DSE_action_space.get_action_list())

            self.reward_array.append(reward)
            self.reward2_array.append(reward2)
            self.reward3_array.append(reward3)

            # compute and record return
            return_g = 0
            T = len(reward_list)
            for t in range(T):
                return_g = reward_list[T - 1 - t] + self.GEMA * return_g
                return_list.append(torch.tensor(return_g).reshape(1))
            return_list.reverse()
            self.worksheet.write(period + 1, 0, period)
            self.worksheet.write(period + 1, 1, return_list[0].item())

            return_g2 = 0
            T2 = len(reward2_list)
            for t in range(T2):
                return_g2 = reward2_list[T2 - 1 - t] + self.GEMA * return_g2
                return2_list.append(torch.tensor(return_g2).reshape(1))
            return2_list.reverse()
            self.worksheet.write(period + 1, 3, return2_list[0].item())

            return_g3 = 0
            T3 = len(reward3_list)
            for t in range(T3):
                return_g3 = reward3_list[T3 - 1 - t] + self.GEMA * return_g3
                return3_list.append(torch.tensor(return_g3).reshape(1))
            return3_list.reverse()
            self.worksheet.write(period + 1, 5, return3_list[0].item())

            sample = {
                "reward": reward,
                "reward2": reward2,
                "reward3": reward3,
                "return_list": return_list,
                "status_list": status_list,
                "action_list": action_list,
            }
            if len(self.replay_buffer) < self.BATCH_SIZE:
                self.replay_buffer.append(sample)
            else:
                min_sample = min(
                    self.replay_buffer, key=lambda sample: sample["reward"]
                )
                if sample["reward"] > min_sample["reward"]:
                    index = self.replay_buffer.index(min_sample)
                    self.replay_buffer[index] = sample
                elif (sample["reward"] == min_sample["reward"]) and (
                    sample["reward3"] > min_sample["reward3"]
                ):
                    index = self.replay_buffer.index(min_sample)
                    self.replay_buffer[index] = sample
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++second++++++++++++++++++++++++++++++++++++

            if len(self.replay_buffer2) < self.BATCH_SIZE:
                self.replay_buffer2.append(sample)
            else:
                min_sample = min(
                    self.replay_buffer2, key=lambda sample: sample["reward2"]
                )
                if sample["reward2"] > min_sample["reward2"]:
                    index = self.replay_buffer2.index(min_sample)
                    self.replay_buffer2[index] = sample
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++third++++++++++++++++++++++++++++++++++++

            if len(self.replay_buffer3) < self.BATCH_SIZE:
                self.replay_buffer3.append(sample)
            else:
                min_sample = min(
                    self.replay_buffer3, key=lambda sample: sample["reward3"]
                )
                if sample["reward3"] > min_sample["reward3"]:
                    index = self.replay_buffer3.index(min_sample)
                    self.replay_buffer3[index] = sample
                elif (sample["reward3"] == min_sample["reward3"]) and (
                    sample["reward"] > min_sample["reward"]
                ):
                    index = self.replay_buffer3.index(min_sample)
                    self.replay_buffer3[index] = sample

            if period < self.SAMPLE_PERIOD_BOUND:
                pass
            elif self.replay_buffer and self.replay_buffer2 and self.replay_buffer3:
                #### compute loss and update actor network
                loss = torch.tensor(0)
                for _ in range(self.BATCH_SIZE):
                    #### random sample trace from replay buffer
                    sample_selected = random.choice(self.replay_buffer)
                    s_return_list = sample_selected["return_list"]
                    s_status_list = sample_selected["status_list"]
                    s_action_list = sample_selected["action_list"]

                    #### compute log_prob and entropy
                    T = len(s_return_list)
                    sample_loss = torch.tensor(0)
                    pi_noise = torch.tensor(1)
                    for t in range(T):
                        s_entropy, s_log_prob = get_log_prob(
                            self.policyfunction,
                            self.DSE_action_space,
                            s_status_list[t],
                            s_action_list[t],
                            t,
                        )
                        retrun_item = (
                            -1 * s_log_prob * (s_return_list[t] - self.BASE_LINE)
                        )
                        entropy_item = -1 * self.ENTROPY_RATIO * s_entropy
                        sample_loss = sample_loss + retrun_item + entropy_item
                        # pi_noise = pi_noise * s_probs_noise_list[t].detach()
                    #### accumulate loss
                    sample_loss = sample_loss / T
                    loss = loss + sample_loss
                loss = loss / self.BATCH_SIZE

                # print(f"loss:{loss}")
                self.worksheet.write(int(period / self.BATCH_SIZE) + 1, 2, loss.item())
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++second++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                loss2 = torch.tensor(0)
                for _ in range(self.BATCH_SIZE):
                    #### random sample trace from replay buffer
                    sample2_selected = random.choice(self.replay_buffer2)
                    s_return_list2 = sample2_selected["return_list"]
                    s_status_list2 = sample2_selected["status_list"]
                    s_action_list2 = sample2_selected["action_list"]

                    #### compute log_prob and entropy
                    T2 = len(s_return_list2)
                    sample2_loss = torch.tensor(0)
                    pi_noise = torch.tensor(1)
                    for t in range(T2):
                        s_entropy2, s_log_prob2 = get_log_prob(
                            self.policyfunction_2,
                            self.DSE_action_space,
                            s_status_list2[t],
                            s_action_list2[t],
                            t,
                        )
                        retrun_item2 = (
                            -1 * s_log_prob2 * (s_return_list2[t] - self.BASE_LINE)
                        )
                        entropy_item2 = -1 * self.ENTROPY_RATIO * s_entropy2
                        sample2_loss = sample2_loss + retrun_item2 + entropy_item2
                    # pi_noise = pi_noise * s_probs_noise_list[t].detach()
                    #### accumulate loss
                    sample2_loss = sample2_loss / T2
                    loss2 = loss2 + sample2_loss
                loss2 = loss2 / self.BATCH_SIZE

                # print(f"loss:{loss}")
                self.worksheet.write(int(period / self.BATCH_SIZE) + 1, 4, loss2.item())
                self.policy_optimizer_2.zero_grad()
                loss2.backward()
                self.policy_optimizer_2.step()

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++third+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                loss3 = torch.tensor(0)
                for _ in range(self.BATCH_SIZE):
                    #### random sample trace from replay buffer
                    sample3_selected = random.choice(self.replay_buffer3)
                    s_return_list3 = sample3_selected["return_list"]
                    s_status_list3 = sample3_selected["status_list"]
                    s_action_list3 = sample3_selected["action_list"]

                    #### compute log_prob and entropy
                    T3 = len(s_return_list3)
                    sample3_loss = torch.tensor(0)
                    pi_noise = torch.tensor(1)
                    for t in range(T3):
                        s_entropy3, s_log_prob3 = get_log_prob(
                            self.policyfunction_3,
                            self.DSE_action_space,
                            s_status_list3[t],
                            s_action_list3[t],
                            t,
                        )
                        retrun_item3 = (
                            -1 * s_log_prob3 * (s_return_list3[t] - self.BASE_LINE)
                        )
                        entropy_item3 = -1 * self.ENTROPY_RATIO * s_entropy3
                        sample3_loss = sample3_loss + retrun_item3 + entropy_item3
                    # pi_noise = pi_noise * s_probs_noise_list[t].detach()
                    #### accumulate loss
                    sample3_loss = sample3_loss / T3
                    loss3 = loss3 + sample3_loss
                loss3 = loss3 / self.BATCH_SIZE

                # print(f"loss:{loss}")
                self.worksheet.write(int(period / self.BATCH_SIZE) + 1, 6, loss3.item())
                self.policy_optimizer_3.zero_grad()
                loss3.backward()
                self.policy_optimizer_3.step()

                # print("loss1:",loss)
                # print("loss2:", loss2)
                # print("loss3:", loss3)

            else:
                print("no avaiable sample")
            period = period + 1
        # end for-period
        self.workbook.save("record/new_reward&return/RLDSE_reward_record_old.xls")
        print(f"stack_renew {stack_renew}")

    def test(self):
        pass
        # for period in range(1):
        # 	for step in range(self.DSE_action_space.get_lenth()):
        #
        # 		action, _ = self.actor.action_choose_with_no_grad_3(self.policyfunction,self.policyfunction_2,self.policyfunction_3, self.DSE_action_space, step,555,is_train = False)
        #
        # 		self.fstatus = self.DSE_action_space.sample_one_dimension(step, action)
        # 	self.evaluation.update_parameter(self.fstatus)
        # 	self.fruntime, t_L = self.evaluation.runtime()
        # 	self.fruntime = self.fruntime * 1000
        # 	self.fpower = self.evaluation.power()
        # 	print(
        # 		"\n@@@@  TEST  @@@@\n",
        # 		"final_status\n", self.fstatus,
        # 		"\nfinal_runtime\n", self.fruntime,
        # 		"\npower\n", self.fpower,
        # 		"\nbest_time\n", self.best_objectvalue_list[-1]
        # 	)


def run(iindex):
    print(f"%%%%TEST{iindex} START%%%%")

    DSE = RLDSE(iindex)
    print(f"DSE scale:{DSE.DSE_action_space.get_scale()}")
    DSE.train()
    DSE.test()

    workbook = xlwt.Workbook(encoding="ascii")
    worksheet = workbook.add_sheet("1")
    worksheet.write(0, 0, "index")
    worksheet.write(0, 1, "runtime")
    for index, best_objectvalue in enumerate(DSE.best_objectvalue_list):
        worksheet.write(index + 1, 0, index + 1)
        worksheet.write(index + 1, 1, best_objectvalue)
    name = (
        "record/objectvalue_double_three_policy/"
        + "_"
        + "MOMPRDSE"
        + "_"
        + "runtime"
        + str(iindex)
        + "500"
        + ".xls"
    )
    workbook.save(name)

    workbook = xlwt.Workbook(encoding="ascii")
    worksheet = workbook.add_sheet("1")
    worksheet.write(0, 0, "index")
    worksheet.write(0, 1, "power")
    for index, best_objectvalue in enumerate(DSE.best_objectvalue2_list):
        worksheet.write(index + 1, 0, index + 1)
        worksheet.write(index + 1, 1, best_objectvalue)
    name = (
        "record/objectvalue_double_three_policy/"
        + "_"
        + "MOMPRDSE"
        + "_"
        + "power"
        + str(iindex)
        + "500"
        + ".xls"
    )
    workbook.save(name)

    workbook2 = xlwt.Workbook(encoding="ascii")
    worksheet2 = workbook2.add_sheet("1")
    worksheet2.write(0, 0, "index")
    worksheet2.write(0, 1, "objectvalue1")
    worksheet2.write(0, 2, "objectvalue2")
    for index, objectvalue in enumerate(DSE.all_objectvalue):
        worksheet2.write(index + 1, 0, index + 1)
        worksheet2.write(index + 1, 1, objectvalue)
    for index, objectvalue in enumerate(DSE.all_objectvalue2):
        worksheet2.write(index + 1, 2, objectvalue)
    name = (
        "record/objectvalue_double_three_policy/"
        + "_"
        + "MOMPRDSE"
        + "_"
        + str(iindex)
        + "all_value"
        + "500"
        + ".xls"
    )
    workbook2.save(name)

    """
	tsne3D(DSE.action_array, DSE.reward_array, "ERDSE" + "_" + DSE.nnmodel + "_" + DSE.target)
	high_value_reward = 0
	for reward in DSE.reward_array:
		if(reward >= 10): high_value_reward += 1
	high_value_reward_proportion = high_value_reward/len(DSE.reward_array)
	hfile = open("high_value_reward_proportion_"+str(iindex)+".txt", "w")
	print(f"@@@@high-value design point proportion:{high_value_reward_proportion}@@@@", file=hfile)
	"""
    print(f"%%%%TEST{iindex} END%%%%")


if __name__ == "__main__":
    USE_MULTIPROCESS = False
    TEST_BOUND = 4

    if USE_MULTIPROCESS:
        iindex_list = list()
        for i in range(TEST_BOUND):
            if i < 10 or i >= 20:
                continue
            iindex_list.append(i)

        pool = Pool(1)
        pool.map(run, iindex_list)
        pool.close()
        pool.join()
    else:
        for iindex in range(3, TEST_BOUND):
            run(iindex)
