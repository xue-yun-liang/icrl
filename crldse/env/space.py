import torch
import numpy
import random
import pdb
import numpy as np
import matplotlib.pyplot as plt

# from logger import Logger
from crldse.env.eval import evaluation_function


class dimension_discrete:
    def __init__(
        self,
        name,
        default_value,
        step,
        rrange,
        frozen=False,
    ):
        """
        "name"-string
        "default_value"-int
        "step"-int
        "rrange"-[low-int,high-int]
        """
        self.name = name
        self.default_value = default_value
        self.current_value = default_value
        self.step = step
        self.frozen = frozen

        assert rrange[0] <= rrange[-1]
        self.rrange = rrange
        self.scale = (self.rrange[-1] - self.rrange[0]) // self.step + 1

        self.default_index = int((default_value - self.rrange[0]) // self.step)
        self.current_index = self.default_index
        """
		sample_box offers the sample space for discrete dimension
		every item in sample_box is a avaiable value for that dimension
		NOW: thr range of sample_box is [rrange[0] ~ rrange[0]+step*(scale-1) rather than [rrange[0] ~ rrange[-1]]
		that's not a big deal
		"""
        self.sample_box = []
        for idx in range(int(self.scale)):
            self.sample_box.append(self.rrange[0] + idx * step)

    def set(self, sample_index):
        assert sample_index >= 0 and sample_index <= self.scale - 1
        self.current_index = sample_index
        self.current_value = self.sample_box[sample_index]

    def original_set(self, sample_index):
        assert sample_index >= 0 and sample_index <= self.scale - 1
        self.default_index = sample_index
        self.default_value = self.sample_box[sample_index]

    def reset(self):
        self.current_index = self.default_index
        self.current_value = self.default_value

    def get_name(self):
        return self.name

    def get_scale(self):
        return self.scale

    def get_range_upbound(self):
        return self.rrange[-1]

    def sample(self):
        if self.frozen == False:
            self.current_value = np.random.choice(self.sample_box, 1, replace=True)[0]
            self.current_index = self.sample_box.index(self.current_value)
        return self.current_value

    def get_current_index(self):
        return self.current_index

    def get_current_value(self):
        return self.current_value

    def get_sample_box(self):
        return self.sample_box

    def froze(self):
        self.frozen = True

    def release(self):
        self.frozen = False

    def check_dimension_discrete(self):
        self.reset()
        print(self.sample_box)
        return

    def get_norm_current_value(self):
        return self.get_current_value() / self.get_range_upbound()


class design_space:
    def __init__(self) -> None:
        """
        dimension_box is a list of dict which is consist of two item,
        "name":str and "dimension":dimension_discrete
        """
        self.dimension_box = []
        self.length = 0
        self.scale = 1

    def append(self, dimension_discrete) -> None:
        self.dimension_box.append(dimension_discrete)
        self.length = self.length + 1
        self.scale = self.scale * dimension_discrete.get_scale()

    def get_status(self) -> dict:
        """
        return a dict  that refer to a status"name":"dimension_value"
        e.g. {'core': 1, 'l1i_size': 1, 'l1d_size': 1, 'l2_size': 6,
        'l1d_assoc': 1, 'l1i_assoc': 1, 'l2_assoc': 1, 'sys_clock': 2}
        """
        status = dict()
        for item in self.dimension_box:
            status[item.get_name()] = item.get_current_value()
        return status

    def get_status_value(self) -> list:
        """only return the value of statue. e.g. [1, 1, 1, 6, 1, 1, 1, 2]
        """
        status_value = list()
        for item in self.dimension_box:
            status_value.append(item.get_current_value())
        return status_value

    def get_action_list(self) -> None:
        """only return the value of action. e.g. [0, 0, 0, 0, 0, 0, 0, 0]
        """
        action_list = list()
        for item in self.dimension_box:
            action_list.append(item.get_current_index())
        return action_list

    def print_status(self) -> None:
        """print the status in dict form"""
        for item in self.dimension_box:
            print(item.get_name(), item.get_current_value())

    def sample_one_dimension(self, dimension_index) -> dict:
        assert dimension_index >= 0 and dimension_index <= self.length - 1
        self.dimension_box[dimension_index].sample()
        return self.get_status()

    def set_one_dimension(self, dimension_index, sample_index) -> dict:
        assert dimension_index >= 0 and dimension_index <= self.length - 1
        self.dimension_box[dimension_index].set(sample_index)
        return self.get_status()

    def set_status(self, best_action_list) -> dict:
        for dimension, action in zip(self.dimension_box, best_action_list):
            dimension.set(action)
        return self.get_status()

    def original_set_status(self, best_action_list) -> dict:
        """set the default_index and default_value"""
        for dimension, action in zip(self.dimension_box, best_action_list):
            dimension.original_set(action)
        return self.get_status()

    def reset_status(self) -> dict:
        """restore status to default index and default values"""
        for dimension in self.dimension_box:
            dimension.reset()
        return self.get_status()

    def get_length(self):
        """the length of dimension_box"""
        return self.length

    def get_scale(self):
        """the total number of design points"""
        return self.scale

    def get_dimension_current_index(self, dimension_index):
        """return a index of given dims"""
        return self.dimension_box[dimension_index].get_current_index()

    def get_dimension_scale(self, dimension_index):
        """return the scale of given dims"""
        return self.dimension_box[dimension_index].get_scale()

    def get_dimension_sample_box(self, dimension_index):
        """return the sample_box of given dims"""
        return self.dimension_box[dimension_index].sample_box

    def froze_one_dimension(self, dimension_index):
        self.dimension_box[dimension_index].froze()

    def release_one_dimension(self, dimension_index):
        self.dimension_box[dimension_index].release()

    def froze_dimension(self, dimension_index_list):
        for index in dimension_index_list:
            self.froze_one_dimension(index)

    def release_dimension(self, dimension_index_list):
        for index in dimension_index_list:
            self.release_one_dimension(index)

    def get_obs(self):
        """the actual value of current status"""
        obs_list = list()
        for item in self.dimension_box:
            obs_list.append(item.get_norm_current_value())
        obs = numpy.array(obs_list)
        return obs


def create_space_crldse():
    DSE_action_space = design_space()

    core = dimension_discrete(
        name="core",
        default_value=1,
        step=1,
        rrange=[1, 16],
    )

    l1i_size = dimension_discrete(
        name="l1i_size",
        default_value=1,
        step=1,
        rrange=[1, 12],
    )
    l1d_size = dimension_discrete(
        name="l1d_size",
        default_value=1,
        step=1,
        rrange=[1, 12],
    )
    l2_size = dimension_discrete(
        name="l2_size",
        default_value=6,
        step=1,
        rrange=[6, 16],
    )
    l1d_assoc = dimension_discrete(
        name="l1d_assoc",
        default_value=1,
        step=1,
        rrange=[1, 4],
    )
    l1i_assoc = dimension_discrete(
        name="l1i_assoc",
        default_value=1,
        step=1,
        rrange=[1, 4],
    )

    l2_assoc = dimension_discrete(
        name="l2_assoc",
        default_value=1,
        step=1,
        rrange=[1, 4],
    )

    sys_clock = dimension_discrete(
        name="sys_clock",
        default_value=2,
        step=0.1,
        rrange=[2, 4],
    )

    DSE_action_space = design_space()
    DSE_action_space.append(core)
    DSE_action_space.append(l1i_size)
    DSE_action_space.append(l1d_size)
    DSE_action_space.append(l2_size)
    DSE_action_space.append(l1d_assoc)
    DSE_action_space.append(l1i_assoc)
    DSE_action_space.append(l2_assoc)
    DSE_action_space.append(sys_clock)

    return DSE_action_space


class environment_erdse:

    def __init__(
        self,
        layer_num,
        model,
        target,
        goal,
        constraints,
        nlabel,
        algo,
        test=False,
        delay_reward=True,
    ):
        self.layer_num = layer_num
        self.model = model
        self.target = target
        self.goal = goal
        self.constraints = constraints
        self.nlabel = nlabel
        self.algo = algo
        self.test = test
        self.delay_reward = delay_reward

        self.best_result = 1e10
        self.best_reward = 0
        self.sample_time = 0

        print(f"Period\tResult", end="\n", file=self.result_log)

        # the next one line actural: self.design_space = create_space_erdse()
        self.design_space = create_space_crldse()

        self.design_space_dimension = self.design_space.get_length()
        self.action_dimension_list = list()
        self.action_limit_list = list()
        for dimension in self.design_space.dimension_box:
            self.action_dimension_list.append(int(dimension.get_scale()))
            self.action_limit_list.append(int(dimension.get_scale() - 1))

        self.evaluation = evaluation_function(self.model, self.target)

    def reset(self):
        self.design_space.status_reset()
        return self.design_space.get_obs()

    def step(self, step, act, deterministic=False):
        if deterministic:
            act = torch.argmax(torch.as_tensor(act).view(-1) if not torch.is_tensor(act) \
                else torch.argmax(act, dim=-1).item(), dim=-1).item()
        else:
            if self.algo == "SAC":
                act = act if torch.is_tensor(act) else torch.as_tensor(act, dtype=torch.float32).view(-1)
                act = torch.softmax(act, dim=-1)
                # sample act based on the probability of the result of act softmax
                act = int(act.multinomial(num_samples=1).data.item())
            elif self.algo == "PPO":
                pass

        current_status = self.design_space.get_status()
        next_status = self.design_space.sample_one_dimension(step, act)
        obs = self.design_space.get_obs()

        if step < (self.design_space.get_length() - 1):
            not_done = 1
        else:
            not_done = 0

        if not_done:
            if self.delay_reward:
                reward = float(0)
            else:
                self.evaluation.update_parameter(next_status, has_memory=True)
                runtime, t_L = self.evaluation.runtime()
                runtime = runtime * 1000
                power, DSP = self.evaluation.power(), self.evaluation.DSP()
                energy = self.evaluation.energy()
                BW = self.evaluation.BW_total
                BRAM = self.evaluation.BRAM_total

                self.constraints.update(
                    {"DSP": DSP, "POWER": power, "BW": BW, "BRAM": BRAM}
                )

                if self.goal == "latency":
                    reward = 1000 / (runtime * self.constraints.get_punishment())
                    result = runtime
                elif self.goal == "energy":
                    reward = 1000 / (energy * self.constraints.get_punishment())
                    result = energy
                elif self.goal == "latency&energy":
                    reward = 1000 / (
                        (runtime * energy) * self.constraints.get_punishment()
                    )
                    result = runtime * energy

                if (result < self.best_result) and self.constraints.is_all_meet():
                    self.best_result = result
                if (reward > self.best_reward) and self.constraints.is_all_meet():
                    self.best_reward = reward

                if not self.test:
                    self.sample_time += 1
                    print(
                        f"{self.sample_time}\t{self.best_result}",
                        end="\n",
                        file=self.result_log,
                    )
        else:
            self.evaluation.update_parameter(next_status, has_memory=True)
            runtime, t_L = self.evaluation.runtime()
            runtime = runtime * 1000
            power, DSP = self.evaluation.power(), self.evaluation.DSP()
            energy = self.evaluation.energy()
            BW = self.evaluation.BW_total
            BRAM = self.evaluation.BRAM_total

            self.constraints.update(
                {"DSP": DSP, "POWER": power, "BW": BW, "BRAM": BRAM}
            )

            if self.goal == "latency":
                reward = 1000 / (runtime * self.constraints.get_punishment())
                result = runtime
            elif self.goal == "energy":
                reward = 1000 / (energy * self.constraints.get_punishment())
                result = energy
            elif self.goal == "latency&energy":
                reward = 1000 / ((runtime * energy) * self.constraints.get_punishment())
                result = runtime * energy

            if (result < self.best_result) and self.constraints.is_all_meet():
                self.best_result = result
            if (reward > self.best_reward) and self.constraints.is_all_meet():
                self.best_reward = reward

            if not self.test:
                self.sample_time += 1
                print(
                    f"{self.sample_time}\t{self.best_result}",
                    end="\n",
                    file=self.result_log,
                )

        done = not not_done

        return obs, reward, done, {}

    def sample(self, step):
        idx = random.randint(0, self.design_space.get_dimension_scale(step) - 1)
        pi = torch.zeros(int(self.design_space.get_dimension_scale(step)))
        pi[idx] = 1
        return pi


if __name__ == "__main__":
    dse_space = create_space_crldse()
    dse_space.print_status()
