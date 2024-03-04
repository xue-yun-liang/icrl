from crldse.space import dimension_discrete
from crldse.space import design_space
from crldse.space import create_space_gem5
from crldse.space import tsne3D
from crldse.actor import actor_e_greedy, actor_policyfunction
from crldse.evaluation import evaluation_function
from crldse.config import my_test_config
from multiprocessing import Pool
from crldse.gem5_mcpat_evaluation import evaluation
import torch
import random
import numpy
import pdb
import copy
import xlwt

debug = False


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
        dimension_index_normalize = dimension_index / self.space_lenth
        input = torch.cat(
            (qfunction_input, torch.tensor(dimension_index_normalize).float().view(1)),
            dim=-1,
        )
        out1 = torch.nn.functional.relu(self.fc1(input))
        out2 = torch.nn.functional.relu(self.fc2(out1))
        out3 = self.fc3[dimension_index](out2)
        return torch.nn.functional.softmax(out3, dim=-1)


class RLDSE:
    def __init__(self, iindex):

        self.iindex = iindex

        seed = self.iindex * 10000
        # atype is reference to models' config
        atype = int(self.iindex / 10)

        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

        #### step1 assign model
        self.config = my_test_config()
        # self.nnmodel = self.config.nnmodel
        # self.layer_num = self.config.layer_num

        #### step2 assign platform
        # constrain
        # self.target = self.config.target
        self.constraints = self.config.constraints

        #### step3 assign goal
        # self.goal = self.config.goal
        self.config.config_check()

        # record the train process
        self.workbook = xlwt.Workbook(encoding="ascii")
        self.worksheet = self.workbook.add_sheet("1")
        self.worksheet.write(0, 0, "period")
        self.worksheet.write(0, 1, "return")
        self.worksheet.write(0, 2, "loss")

        ## initial DSE_action_space
        self.DSE_action_space = create_space_gem5()

        # define the hyperparameters
        self.SAMPLE_PERIOD_BOUND = 1
        self.GEMA = 0.999  # RL parameter, discount ratio
        self.ALPHA = 0.001  # RL parameter, learning step rate
        self.THRESHOLD_RATIO = 2  # 0.05
        self.BATCH_SIZE = 1
        self.BASE_LINE = 0
        self.ENTROPY_RATIO = 0
        self.PERIOD_BOUND = 500

        # initial mlp_policyfunction, every action dimension owns a policyfunction
        # TODO:share the weight of first two layer among policyfunction
        action_scale_list = list()
        for dimension in self.DSE_action_space.dimension_box:
            action_scale_list.append(int(dimension.get_scale()))
        self.policyfunction = mlp_policyfunction(
            self.DSE_action_space.get_lenth(), action_scale_list
        )

        ##initial e_greedy_policy_function
        self.actor = actor_policyfunction()

        ##initial evaluation
        # self.evaluation = evaluation_function(self.nnmodel, self.target)

        ##initial optimizer
        self.policy_optimizer = torch.optim.Adam(
            self.policyfunction.parameters(),
            lr=self.ALPHA,
        )

        #### loss replay buffer, in order to record and reuse high return trace
        self.loss_buffer = list()

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

    def train(self):
        current_status = dict()  # S
        next_status = dict()  # S'

        loss = torch.tensor(0)  # define loss function
        batch_index = 0

        period_bound = self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND
        for period in range(self.PERIOD_BOUND):
            logger.info(f"period:{period}", end="\r")
            # here may need a initial function for action_space
            self.DSE_action_space.status_reset()

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
                    metrics = evaluation(next_status)
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
                        logger.info(f"best_status:{objectvalue}")
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
            self.worksheet.write(period + 1, 0, period)
            self.worksheet.write(period + 1, 1, return_list[0].item())

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


def run(iindex):
    logger.info(f"%%%%TEST{iindex} START%%%%")
    DSE = RLDSE(iindex)
    DSE.train()
    DSE.test()

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
        "record/objectvalue/"
        + "REINFORCE"
        + "_"
        + "gem5"
        + "_"
        + str(iindex)
        + "all_value"
        + ".xls"
    )
    workbook2.save(name)

    """
	tsne3D(DSE.action_array, DSE.reward_array, "RI" + "_" + DSE.nnmodel + "_" + DSE.target)
	high_value_reward = 0
	for reward in DSE.reward_array:
		if(reward >= 10): high_value_reward += 1
	high_value_reward_proportion = high_value_reward/len(DSE.reward_array)
	hfile = open("high_value_reward_proportion_"+str(iindex)+"_"+ "ACDSE" +".txt", "w")
	logger.info(f"@@@@high-value design point proportion:{high_value_reward_proportion}@@@@", file=hfile)
	"""

    logger.info(f"%%%%TEST{iindex} END%%%%")


if __name__ == "__main__":
    USE_MULTIPROCESS = False
    TEST_BOUND = 4

    if USE_MULTIPROCESS:
        iindex_list = list()
        for i in range(TEST_BOUND):
            iindex_list.append(i)

        pool = Pool(30)
        pool.map(run, iindex_list)
        pool.close()
        pool.join()
    else:
        for iindex in range(3, TEST_BOUND):
            run(iindex)
