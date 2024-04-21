import yaml

class constraint:
    def __init__(self, name, threshold, threshold_ratio):
        self.name = name
        self.threshold = threshold
        self.threshold_ratio = threshold_ratio

        self.value = 0
        self.margin = (self.threshold - self.value) / self.threshold
        self.stable_margin = self.margin
        self.is_meet_flag = False
        self.l = 0
        self.punishment = 0     # reward factor

    def update(self, value):
        self.value = value
        self.margin = (self.threshold - self.value) / self.threshold
        if self.margin < 0:
            self.is_meet_flag = False
            self.l = self.threshold_ratio
        else:
            self.is_meet_flag = True
            self.l = 0
        self.punishment = (self.value / self.threshold) ** self.l

    def update_stable_margin(self):
        self.stable_margin = self.margin

    def get_name(self):
        return self.name

    def get_threshold(self):
        return self.threshold

    def get_margin(self):
        return self.margin

    def get_stable_margin(self):
        return self.stable_margin

    def is_meet(self):
        return self.is_meet_flag

    def get_punishment(self):
        return self.punishment

    def print(self):
        print(f"name = {self.name}, threshold = {self.threshold}, \
            value = {self.value},margin = {self.margin}, \
            is_meet:{self.is_meet_flag}, punishment = {self.punishment}, \
            stable_margin = {self.stable_margin}")


class constraints:
    def __init__(self):
        self.constraint_list = list()

    def append(self, constraint):
        self.constraint_list.append(constraint)

    def update(self, value_dict):
        assert len(value_dict.items()) == len(self.constraint_list)
        for key, value, constrain in zip(
            value_dict.keys(), value_dict.values(), self.constraint_list
        ):
            assert key == constrain.get_name()
            constrain.update(value)

    def update_stable_margin(self):
        for constraint in self.constraint_list:
            constraint.update_stable_margin()

    def get_stable_margin(self, name_str):
        stable_margin = 0
        for constraint in self.constraint_list:
            if constraint.get_name() == name_str:
                stable_margin = constraint.get_stable_margin()
        return stable_margin

    def is_any_meet(self):
        self.is_any_meet_flag = False
        for constraint in self.constraint_list:
            self.is_any_meet_flag = self.is_any_meet_flag or constraint.is_meet()
        return self.is_any_meet_flag

    def is_all_meet(self):
        self.is_all_meet_flag = True
        for constraint in self.constraint_list:
            self.is_all_meet_flag = self.is_all_meet_flag and constraint.is_meet()
        return self.is_all_meet_flag

    def get_punishment(self):
        self.punishment = 1
        for constraint in self.constraint_list:
            self.punishment = self.punishment * constraint.get_punishment()
        return self.punishment

    def print(self):
        for constraint in self.constraint_list:
            constraint.print()
        
        
class create_constraints_conf():
    def __init__(self, config_data) -> None:
        """
        Generate a constraints class based on the given parameters to control 
        the agent's search process.
        Parameter:
            config_data: a config file's content which store specific threshold values for each mode
        """
        self.target = config_data['target']
        self.goal = config_data['goal']
        self.th_ratio = config_data['th_ratio']
        self.constraints = constraints()

        for constraint_name, value in config_data['constraints'][self.target].items():
            self.constraints.append(constraint(name=constraint_name, threshold=value, threshold_ratio=self.th_ratio))


def print_config(constraints_conf):
    print(f"--------------Config Check--------------")
    print(f"target:{constraints_conf.target:>5}")
    for constraint_ in constraints_conf.constraints.constraint_list:
        print(f"{constraint_.get_name():<5}{constraint_.get_threshold():>5}")
    print(f"th_ratio:{constraints_conf.th_ratio}")
    print(f"goal:{constraints_conf.goal}")
    
    

if __name__ == "__main__":
    with open('./env/config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)
    test_conf = create_constraints_conf(config_data=config_data)
    print_config(test_conf)