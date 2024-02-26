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
        self.punishment = 0

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
        print(
            f"name = {self.name}, threshold = {self.threshold}, value = {self.value}, margin = {self.margin}, is_meet:{self.is_meet_flag}, punishment = {self.punishment}, stable_margin = {self.stable_margin}"
        )


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


class test_config:
    def __init__(self, atype):
        #### step1 define model
        if atype == 0:
            self.nnmodel, self.target = "VGG16", "cloud"
        elif atype == 1:
            self.nnmodel, self.target = "VGG16", "normal"
        elif atype == 2:
            self.nnmodel, self.target = "VGG16", "embed"
        elif atype == 3:
            self.nnmodel, self.target = "MobilenetV3", "cloud"
        elif atype == 4:
            self.nnmodel, self.target = "MobilenetV3", "normal"
        elif atype == 5:
            self.nnmodel, self.target = "MobilenetV3", "embed"

        """
		# self.nnmodel = "VGG16"
		# self.nnmodel = "VGG19"
		# self.nnmodel = "ENASNN"
		self.nnmodel = "MobilenetV3"

		#### step2 define platform and constrain
		# self.target = "normal"
		self.target = "cloud"
		# self.target = "embed"
		"""

        #### step3 define goal
        # self.goal = "latency"
        self.goal = "energy"
        # self.goal = "latency&energy"

        #### final we can define the constrain
        if self.nnmodel == "VGG16":
            self.layer_num = 21
        elif self.nnmodel == "VGG19":
            self.layer_num = 24
        elif self.nnmodel == "ENASNN":
            self.layer_num = 30
        elif self.nnmodel == "MobilenetV3":
            self.layer_num = 76

        if self.target == "normal":  # ref vertex7
            self.DSP_THRESHOLD = 2800
            self.POWER_THRESHOLD = 60
            self.BW_THRESHOLD = 205  # Gbps actually is 204.8
            self.BRAM_THRESHOLD = 37  # Mb
        elif self.target == "cloud":  # ref vertex ultrascale+
            self.DSP_THRESHOLD = 6900
            self.POWER_THRESHOLD = 500
            self.BW_THRESHOLD = 385  # Gbps actually is 384
            self.BRAM_THRESHOLD = 345.9  # Mb
        elif self.target == "embed":  # ref zynq7000
            self.DSP_THRESHOLD = 900
            # self.POWER_THRESHOLD = 5 #### in big space which contains memory, 5w is to small and almost no point can meet the constraints
            self.POWER_THRESHOLD = 20
            self.BW_THRESHOLD = 103  # Gbps actually is 102.4
            self.BRAM_THRESHOLD = 4.9  # Mb
            # self.BRAM_THRESHOLD = 10#Mb

        self.THRESHOLD_RATIO = 2
        DSP = constraint(
            name="DSP",
            threshold=self.DSP_THRESHOLD,
            threshold_ratio=self.THRESHOLD_RATIO,
        )
        POWER = constraint(
            name="POWER",
            threshold=self.POWER_THRESHOLD,
            threshold_ratio=self.THRESHOLD_RATIO,
        )
        BW = constraint(
            name="BW", threshold=self.BW_THRESHOLD, threshold_ratio=self.THRESHOLD_RATIO
        )
        BRAM = constraint(
            name="BRAM",
            threshold=self.BRAM_THRESHOLD,
            threshold_ratio=self.THRESHOLD_RATIO,
        )
        self.constraints = constraints()
        self.constraints.append(DSP)
        # self.constraints.append(POWER)
        self.constraints.append(BW)
        self.constraints.append(BRAM)

    def config_check(self):
        print(f"######Config Check######")
        print(f"configtype:test")
        print(f"nnmodel:{self.nnmodel}")
        print(f"layer_num:{self.layer_num}")
        print(f"target:{self.target}")
        for constraint in self.constraints.constraint_list:
            print(f"{constraint.get_name()}:{constraint.get_threshold()}")
        print(f"goal:{self.goal}")


class my_test_config:
    def __init__(self):
        self.AREA_THRESHOLD = 411
        self.THRESHOLD_RATIO = 2
        AREA = constraint(
            name="AREA",
            threshold=self.AREA_THRESHOLD,
            threshold_ratio=self.THRESHOLD_RATIO,
        )
        self.constraints = constraints()
        self.constraints.append(AREA)

    def config_check(self):
        print(f"######Config Check######")
        print(f"configtype:test")
        print(f"AERA:{self.AREA_THRESHOLD}")


class my_test_config_2:
    def __init__(self):
        self.AREA_THRESHOLD = 165
        self.THRESHOLD_RATIO = 2
        AREA = constraint(
            name="AREA",
            threshold=self.AREA_THRESHOLD,
            threshold_ratio=self.THRESHOLD_RATIO,
        )
        self.constraints = constraints()
        self.constraints.append(AREA)

    def config_check(self):
        print(f"######Config Check######")
        print(f"configtype:test")
        print(f"AERA:{self.AREA_THRESHOLD}")


#### only used for test6, test15, test16, test17
class debug_config:
    def __init__(self, atype):
        #### step1 define model
        if atype == 0:
            self.nnmodel, self.target = "VGG16", "cloud"
        elif atype == 1:
            self.nnmodel, self.target = "VGG16", "normal"
        elif atype == 2:
            self.nnmodel, self.target = "VGG16", "embed"
        elif atype == 3:
            self.nnmodel, self.target = "MobilenetV3", "cloud"
        elif atype == 4:
            self.nnmodel, self.target = "MobilenetV3", "normal"
        elif atype == 5:
            self.nnmodel, self.target = "MobilenetV3", "embed"

        """
		self.nnmodel = "VGG16"
		#self.nnmodel = "VGG19"
		#self.nnmodel = "ENASNN"
		#self.nnmodel = "MobilenetV3"

		#### step2 define platform and constrain
		#self.target = "normal"
		#self.target = "cloud"
		self.target = "embed"
		"""

        #### step3 define goal
        self.goal = "latency"
        # self.goal = "energy"
        # self.goal = "latency&energy"

        #### final we can define the constrain
        if self.nnmodel == "VGG16":
            self.layer_num = 21
        elif self.nnmodel == "VGG19":
            self.layer_num = 24
        elif self.nnmodel == "ENASNN":
            self.layer_num = 30
        elif self.nnmodel == "MobilenetV3":
            self.layer_num = 76

        if self.target == "normal":  # ref vertex7
            self.DSP_THRESHOLD = 2800
            self.POWER_THRESHOLD = 60
            self.BW_THRESHOLD = 204.8  # Gbps
            self.BRAM_THRESHOLD = 37  # Mb
        elif self.target == "cloud":  # ref vertex ultrascale+
            self.DSP_THRESHOLD = 6900
            self.POWER_THRESHOLD = 500
            self.BW_THRESHOLD = 384  # Gbps
            self.BRAM_THRESHOLD = 345.9  # Mb
        elif self.target == "embed":  # ref zynq7000
            self.DSP_THRESHOLD = 900
            # self.POWER_THRESHOLD = 5
            self.POWER_THRESHOLD = 20
            self.BW_THRESHOLD = 102.4  # Gbps
            self.BRAM_THRESHOLD = 4.9  # Mb

        self.THRESHOLD_RATIO = 2
        DSP = constraint(
            name="DSP",
            threshold=self.DSP_THRESHOLD,
            threshold_ratio=self.THRESHOLD_RATIO,
        )
        POWER = constraint(
            name="POWER",
            threshold=self.POWER_THRESHOLD,
            threshold_ratio=self.THRESHOLD_RATIO,
        )
        BW = constraint(
            name="BW",
            threshold=self.DSP_THRESHOLD,
            threshold_ratio=self.THRESHOLD_RATIO,
        )
        BRAM = constraint(
            name="BRAM",
            threshold=self.DSP_THRESHOLD,
            threshold_ratio=self.THRESHOLD_RATIO,
        )
        self.constraints.append(DSP)
        self.constraints.append(POWER)
        self.constraints.append(BW)
        self.constraints.append(BRAM)

    def config_check(self):
        print(f"######Config Check######")
        print(f"configtype:debug")
        print(f"nnmodel:{self.nnmodel}")
        print(f"layer_num:{self.layer_num}")
        print(f"target:{self.target}")
        for constraint in self.constraints.constraint_list:
            print(f"{constraint.get_name()}:{constraint.get_threshold()}")
        print(f"goal:{self.goal}")
