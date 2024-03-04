####version2021.1.14####
####repair the computing of pro_time and power, which correct the IO computing with considerring bandwidth

import math
import os
import pdb


class evaluation_function:
    def __init__(self, nnmodel, target):
        # TODO parameters_dict will be used for rapid parameters matching
        # self.network_parameters_dict = dict()
        # self.accelerator_parameters_dict = dict()
        assert (nnmodel is not None) or (target is not None)
        self.nnmodel = nnmodel
        if self.nnmodel == "VGG16":
            infile_network = "VGG16.txt"
            infile_accelerator = "VGG16_mapping.txt"
        elif self.nnmodel == "VGG19":
            infile_network = "VGG19.txt"
            infile_accelerator = "VGG19_mapping.txt"
        elif self.nnmodel == "ENASNN":
            infile_network = "squeeze.txt"
            infile_accelerator = "squeeze_mapping.txt"
        elif self.nnmodel == "MobilenetV3":
            infile_network = "MobilenetV3.txt"
            infile_accelerator = "MobilenetV3_mapping.txt"

        """
        self.target = target
        if(self.target == "normal"):#ref vertex7, DDR3-1600-64bit-2channel
            self.BW_total = 204.8#Gbps
            self.BRAM_total = 37#Mb
        elif(self.target == "cloud"):#ref vertex ultrascale+, DDR4-2400-80bit-2channel
            self.BW_total = 384#Gbps
            self.BRAM_total = 345.9#Mb
        elif(self.target == "embed"):#ref zynq7000, DDR3-1600-64bit-1channel
            self.BW_total = 102.4#Gbps
            self.BRAM_total = 4.9#Mb
        """

        k_network = 13
        k_accelerator = 6
        self.read_network(infile_network, k_network)
        self.read_mapping(infile_accelerator, k_accelerator)

        self.Emem_D_list = [8.13e-09, 1.63e-08, 3.25e-08]
        self.Emem_B_list = [2.00e-10, 4.00e-10, 8.00e-10]
        self.Ecpt_list = [3.00e-11, 5.00e-11, 9.00e-11]

        # self.PE_num_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.BW_frequency_list = [1.333, 1.6, 1.866, 2.133, 2.4]
        self.BRAM_list = [
            0.0625,
            0.125,
            0.25,
            0.5,
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
        ]

    def load_data_from_file(self, infile, k):
        f = open(infile, "r")
        sourceInLine = f.readlines()
        dataset = []
        for line in sourceInLine:
            temp1 = line.strip("\n")
            temp2 = temp1.split("\t")
            dataset.append(temp2)
        for i in range(0, len(dataset)):
            for j in range(k):
                dataset[i].append(float(dataset[i][j]))
            del dataset[i][0:k]
        return dataset

    def read_network(self, infile, k):
        # read network parameters from file
        data = self.load_data_from_file(infile, k)

        # draw network parameters
        self.LN = int(data[0][0])

        # type of layer
        # 0=conv. 1=pooling 2=FC.
        self.layer_type = [data[i][1] for i in range(0, self.LN)]

        # input Fmaps
        self.R = [data[i][3] + data[i][10] for i in range(0, self.LN)]
        self.N_in = [data[i][4] for i in range(0, self.LN)]
        self.P = [data[i][10] for i in range(0, self.LN)]  # padding

        # output Fmaps
        self.S = [data[i][5] for i in range(0, self.LN)]
        self.N_out = [data[i][6] for i in range(0, self.LN)]

        # kernel
        self.K = [data[i][7] for i in range(0, self.LN)]
        self.Q = [data[i][8] for i in range(0, self.LN)]
        self.L = [data[i][9] for i in range(0, self.LN)]  # stride

        # BN
        # 0=without BN, 1=with BN
        self.BN = [data[i][11] for i in range(0, self.LN)]

        # bitwidth
        self.bitwidth = int(data[0][12])

    def read_mapping(self, infile, k):
        # read accelerator parameters from file
        data2 = self.load_data_from_file(infile, k)

        self.PE_num = int(data2[0][1])
        self.PE_size = int(data2[0][2])

        self.datain_port_num = data2[0][3]
        self.weightin_port_num = data2[0][4]
        self.dataout_port_num = data2[0][5]

        self.f = data2[1][0]
        self.P_sta = data2[1][1]
        # FC_batch=data2[self.LN+1][0]

        # Unit energy consumption
        # case bitwidth
        self.Emem_D = data2[2][int(math.log(self.bitwidth / 8, 2))]
        self.Emem_B = data2[2][int(math.log(self.bitwidth / 8, 2)) + 3]

        self.Ecpt = data2[3][int(math.log(self.bitwidth / 8, 2))]

        # self.DSP_total=data2[3][3]
        # self.BW_total=data2[3][4]
        self.minibatch = data2[3][5]

        self.scheme_chosen = list()

    def update_parameter(self, status, has_memory=False):
        # TODO: reconsturct the parameter class that we can automativly update those parameters from design_space
        function_test = False
        if function_test:
            self.PE_num = status["PE_num"]
            self.PE_size = status["PE_size"]
            self.f = status["f"] * 1e08
            self.minibatch = status["minibatch"]
            self.bitwidth = (
                2 ** status["bitwidth"] * 8
            )  #### "0" for bit8, "1" for bit16, "2" for bit32
            assert (
                (self.bitwidth == 8) or (self.bitwidth == 16) or (self.bitwidth == 32)
            )
            self.scheme_chosen = list()
            for index in range(self.LN):
                # self.scheme_chosen.append(status[("scheme_chosen" + str(index + 1))])
                self.scheme_chosen.append(3)
        else:
            self.PE_x = status["PE_x"]
            self.PE_y = status["PE_y"]
            self.PE_num = self.PE_x * self.PE_y
            # self.PE_size = status["PE_size"]
            self.f = status["f"] * 1e08
            self.minibatch = status["minibatch"]
            self.bitwidth = (
                2 ** status["bitwidth"] * 8
            )  #### "0" for bit8, "1" for bit16, "2" for bit32
            assert (
                (self.bitwidth == 8) or (self.bitwidth == 16) or (self.bitwidth == 32)
            )

            if has_memory:
                BW_frequency_index = status["BW_frequency"]
                assert (BW_frequency_index > 0) or (BW_frequency_index < 6)
                BW_frequency = self.BW_frequency_list[int(BW_frequency_index)]
                BW_bitwidth = status["BW_bitwidth"]
                BW_channel = status["BW_channel"]
                self.BW_total = BW_frequency * BW_bitwidth * BW_channel
                BRAM_index = status["BRAM_total"]
                assert (BRAM_index > 0) or (BRAM_index < 14)
                self.BRAM_total = self.BRAM_list[int(BRAM_index)]

            self.scheme_chosen = list()
            for index in range(self.LN):
                self.scheme_chosen.append(status[("scheme_chosen" + str(index + 1))])

        self.Emem_D = self.Emem_D_list[int(math.log(self.bitwidth / 8, 2))]
        self.Emem_B = self.Emem_B_list[int(math.log(self.bitwidth / 8, 2))]
        self.Ecpt = self.Ecpt_list[int(math.log(self.bitwidth / 8, 2))]

        # pdb.set_trace()

    def compute_scheme_chosen(self):
        #####################scheme choose####################
        ##########scheme1#############
        datain_scheme1 = []
        dataout_scheme1 = []
        dataweight_scheme1 = []

        IO_scheme1_L = []
        IO_total_scheme1 = 0
        ##########scheme2#############
        datain_scheme2 = []
        dataout_scheme2 = []
        dataout_scheme2_psumout = []
        dataweight_scheme2 = []

        IO_scheme2_L = []
        IO_scheme2_psumout_L = []
        IO_total_scheme2 = 0
        ##########scheme3#############
        datain_scheme3 = []
        dataout_scheme3 = []
        dataweight_scheme3 = []
        dataweight_scheme3_weightin = []

        IO_scheme3_L = []
        IO_scheme3_weightin_L = []
        IO_total_scheme3 = 0
        ###############
        # self.BRAM_total=37
        ################
        BRAM_cpt_req_scheme1_L = []
        BRAM_BN_req_scheme1_L = []
        BRAM_req_scheme1_L = []
        ################
        BRAM_cpt_req_scheme2_L = []
        BRAM_cpt_req_scheme2_psumout_L = []
        BRAM_BN_req_scheme2_L = []
        BRAM_req_scheme2_L = []
        BRAM_req_scheme2_psumout_L = []
        ################
        BRAM_cpt_req_scheme3_L = []
        BRAM_cpt_req_scheme3_weightin_L = []
        BRAM_BN_req_scheme3_L = []
        BRAM_req_scheme3_L = []
        BRAM_req_scheme3_weightin_L = []

        #######output parameter#####
        scheme_chosen = []
        scheme2_psumout = []
        scheme3_weightin = []

        datain = []
        dataout = []
        dataweight = []

        BRAM_cpt_req_L = []
        BRAM_BN_req_L = []
        BRAM_req_L = []

        for i in range(0, self.LN):

            ##########scheme1 IO##########
            dataout_scheme1.append(self.S[i] ** 2 * self.N_out[i])

            if self.layer_type[i] == 1:  # pooling
                datain_scheme1.append(self.R[i] ** 2 * self.N_in[i])
                dataweight_scheme1.append(0)
            else:
                datain_scheme1.append(self.R[i] ** 2 * self.N_in[i] * self.N_out[i])
                dataweight_scheme1.append(self.K[i] ** 2 * self.Q[i])

            IO_scheme1_L.append(
                datain_scheme1[i] + dataout_scheme1[i] + dataweight_scheme1[i]
            )

            ##########scheme2 IO##########
            datain_scheme2.append(self.R[i] ** 2 * self.N_in[i])

            if self.layer_type[i] == 1:  # pooling
                dataweight_scheme2.append(0)
                dataout_scheme2.append(self.S[i] ** 2 * self.N_out[i])
                dataout_scheme2_psumout.append(self.S[i] ** 2 * self.N_out[i])
            else:
                dataweight_scheme2.append(self.K[i] ** 2 * self.Q[i])
                dataout_scheme2.append(self.S[i] ** 2 * self.N_in[i] * self.N_out[i])
                dataout_scheme2_psumout.append(
                    self.S[i] ** 2 * self.N_in[i] * self.N_out[i] * 2
                )

            IO_scheme2_L.append(
                datain_scheme2[i] + dataout_scheme2[i] + dataweight_scheme2[i]
            )
            IO_scheme2_psumout_L.append(
                datain_scheme2[i] + dataout_scheme2_psumout[i] + dataweight_scheme2[i]
            )

            ##########scheme3 IO##########
            datain_scheme3.append(self.R[i] ** 2 * self.N_in[i])
            dataout_scheme3.append(self.S[i] ** 2 * self.N_out[i])

            if self.layer_type[i] == 1:  # pooling
                dataweight_scheme3.append(0)
                dataweight_scheme3_weightin.append(0)

            else:
                dataweight_scheme3.append(self.K[i] ** 2 * self.Q[i] * self.N_out[i])
                dataweight_scheme3_weightin.append(self.K[i] ** 2 * self.Q[i])

            IO_scheme3_L.append(
                datain_scheme3[i] + dataout_scheme3[i] + dataweight_scheme3[i]
            )
            IO_scheme3_weightin_L.append(
                datain_scheme3[i] + dataout_scheme3[i] + dataweight_scheme3_weightin[i]
            )
            ##########scheme1 BRAM###############################

            if self.layer_type[i] == 0:  # conv
                BRAM_cpt_req_scheme1_L.append(
                    (
                        (self.R[i] * (self.K[i] + 1) * 2 + self.K[i] ** 2 * 2)
                        * self.PE_num
                        + self.S[i] ** 2
                    )
                    * self.bitwidth
                    / 1e6
                )
            elif self.layer_type[i] == 1:  # pooling
                BRAM_cpt_req_scheme1_L.append(
                    (self.R[i] * 4 * (self.PE_num + 1) + self.S[i] * self.PE_num)
                    * self.bitwidth
                    / 1e6
                )
            else:
                BRAM_cpt_req_scheme1_L.append(
                    (self.datain_port_num * 2) * self.bitwidth / 1e6
                )

            if self.BN[i] == 1:
                BRAM_BN_req_scheme1_L.append(
                    (self.S[i] ** 2 * 4 / self.minibatch) * self.bitwidth / 1e6
                )
            else:
                BRAM_BN_req_scheme1_L.append(0)

            BRAM_req_scheme1_L.append(
                BRAM_cpt_req_scheme1_L[i] + BRAM_BN_req_scheme1_L[i]
            )

            ##########scheme2 BRAM###############################

            if self.layer_type[i] == 0:  # conv
                BRAM_cpt_req_scheme2_L.append(
                    (
                        self.R[i] ** 2 * 2
                        + self.K[i] ** 2 * 2 * self.PE_num
                        + self.S[i] ** 2 * self.N_out[i]
                    )
                    * self.bitwidth
                    / 1e6
                )
                BRAM_cpt_req_scheme2_psumout_L.append(
                    (
                        self.R[i] ** 2 * 2
                        + self.K[i] ** 2 * 2 * self.PE_num
                        + self.S[i] * self.N_out[i]
                    )
                    * self.bitwidth
                    / 1e6
                )
            elif self.layer_type[i] == 1:  # pooling
                BRAM_cpt_req_scheme2_L.append(
                    (self.R[i] * 4 * (self.PE_num + 1) + self.S[i] * self.PE_num)
                    * self.bitwidth
                    / 1e6
                )
                BRAM_cpt_req_scheme2_psumout_L.append(
                    (self.R[i] * 4 * (self.PE_num + 1) + self.S[i] * self.PE_num)
                    * self.bitwidth
                    / 1e6
                )
            else:  # FC
                BRAM_cpt_req_scheme2_L.append(
                    (self.datain_port_num * 2) * self.bitwidth / 1e6
                )
                BRAM_cpt_req_scheme2_psumout_L.append(
                    (self.datain_port_num * 2) * self.bitwidth / 1e6
                )

            if self.BN[i] == 1:
                BRAM_BN_req_scheme2_L.append(
                    (self.S[i] ** 2 * 4 / self.minibatch) * self.bitwidth / 1e6
                )
            else:
                BRAM_BN_req_scheme2_L.append(0)

            BRAM_req_scheme2_L.append(
                BRAM_cpt_req_scheme2_L[i] + BRAM_BN_req_scheme2_L[i]
            )
            BRAM_req_scheme2_psumout_L.append(
                BRAM_cpt_req_scheme2_psumout_L[i] + BRAM_BN_req_scheme2_L[i]
            )

            ##########scheme3 BRAM###############################

            if self.layer_type[i] == 0:  # conv
                BRAM_cpt_req_scheme3_L.append(
                    (
                        self.R[i] * (self.K[i] + 1) * 2 * self.N_in[i]
                        + self.K[i] ** 2 * 2 * self.N_in[i]
                        + self.S[i] * 1
                    )
                    * self.bitwidth
                    / 1e6
                )
                BRAM_cpt_req_scheme3_weightin_L.append(
                    (
                        self.R[i] * (self.K[i] + 1) * 2 * self.N_in[i]
                        + self.K[i] ** 2 * self.N_out[i] * self.N_in[i]
                        + self.S[i] * 1
                    )
                    * self.bitwidth
                    / 1e6
                )
            elif self.layer_type[i] == 1:  # pooling
                BRAM_cpt_req_scheme3_L.append(
                    (self.R[i] * 4 * (self.PE_num + 1) + self.S[i] * self.PE_num)
                    * self.bitwidth
                    / 1e6
                )
                BRAM_cpt_req_scheme3_weightin_L.append(
                    (self.R[i] * 4 * (self.PE_num + 1) + self.S[i] * self.PE_num)
                    * self.bitwidth
                    / 1e6
                )
            else:  # FC
                BRAM_cpt_req_scheme3_L.append(
                    (self.datain_port_num * 2) * self.bitwidth / 1e6
                )
                BRAM_cpt_req_scheme3_weightin_L.append(
                    (self.datain_port_num * 2) * self.bitwidth / 1e6
                )

            if self.BN[i] == 1:
                BRAM_BN_req_scheme3_L.append(
                    (self.S[i] ** 2 * 4 / self.minibatch) * self.bitwidth / 1e6
                )
            else:
                BRAM_BN_req_scheme3_L.append(0)

            BRAM_req_scheme3_L.append(
                BRAM_cpt_req_scheme3_L[i] + BRAM_BN_req_scheme3_L[i]
            )
            BRAM_req_scheme3_weightin_L.append(
                BRAM_cpt_req_scheme3_weightin_L[i] + BRAM_BN_req_scheme3_L[i]
            )

        #################choose###################################
        #### Fengkaijie-20201029: now we use searching result self.scheme_chosen to replace
        #### computing result scheme_chosen
        for i in range(0, self.LN):
            if BRAM_req_scheme3_weightin_L[i] < self.BRAM_total:
                # scheme_chosen.append(3)
                scheme3_weightin.append(1)
                scheme2_psumout.append(0)

            elif (
                IO_scheme1_L[i] < IO_scheme2_L[i] and IO_scheme1_L[i] < IO_scheme3_L[i]
            ):  # scheme1 IO best
                # scheme_chosen.append(1)
                scheme2_psumout.append(0)
                scheme3_weightin.append(0)

            elif (
                IO_scheme2_L[i] < IO_scheme1_L[i] and IO_scheme2_L[i] < IO_scheme3_L[i]
            ):  # scheme2 IO best
                if BRAM_req_scheme2_L[i] < self.BRAM_total:  # && scheme2 BRAM enough
                    # scheme_chosen.append(2)
                    scheme2_psumout.append(0)
                    scheme3_weightin.append(0)

                elif (
                    IO_scheme2_psumout_L[i] < IO_scheme1_L[i]
                    and IO_scheme2_psumout_L[i] < IO_scheme3_L[i]
                ):  # scheme2 psumout IO best
                    # scheme_chosen.append(2)
                    scheme2_psumout.append(1)
                    scheme3_weightin.append(0)

                elif IO_scheme1_L[i] < IO_scheme3_L[i]:
                    # scheme_chosen.append(1)
                    scheme2_psumout.append(0)
                    scheme3_weightin.append(0)
                else:
                    # scheme_chosen.append(3)
                    scheme2_psumout.append(0)
                    scheme3_weightin.append(0)

            else:  # IO_scheme3_L[i]<IO_scheme1_L[i] and IO_scheme3_L[i]<IO_scheme2_L[i]: #scheme3 IO best
                # scheme_chosen.append(3)
                scheme3_weightin.append(0)
                scheme2_psumout.append(0)

            if self.scheme_chosen[i] == 1:
                datain.append(datain_scheme1[i])
                dataout.append(dataout_scheme1[i])
                dataweight.append(dataweight_scheme1[i])

                BRAM_cpt_req_L.append(BRAM_cpt_req_scheme1_L[i])
                BRAM_BN_req_L.append(BRAM_BN_req_scheme1_L[i])
                BRAM_req_L.append(BRAM_req_scheme1_L[i])

            elif self.scheme_chosen[i] == 2 and scheme2_psumout[i] == 1:
                datain.append(datain_scheme2[i])
                dataout.append(dataout_scheme2_psumout[i])
                dataweight.append(dataweight_scheme2[i])

                BRAM_cpt_req_L.append(BRAM_cpt_req_scheme2_psumout_L[i])
                BRAM_BN_req_L.append(BRAM_BN_req_scheme2_L[i])
                BRAM_req_L.append(BRAM_req_scheme2_psumout_L[i])

            elif self.scheme_chosen[i] == 2 and scheme2_psumout[i] == 0:
                datain.append(datain_scheme2[i])
                dataout.append(dataout_scheme2[i])
                dataweight.append(dataweight_scheme2[i])

                BRAM_cpt_req_L.append(BRAM_cpt_req_scheme2_L[i])
                BRAM_BN_req_L.append(BRAM_BN_req_scheme2_L[i])
                BRAM_req_L.append(BRAM_req_scheme2_L[i])

            elif self.scheme_chosen[i] == 3 and scheme3_weightin[i] == 1:
                datain.append(datain_scheme3[i])
                dataout.append(dataout_scheme3[i])
                dataweight.append(dataweight_scheme3_weightin[i])

                BRAM_cpt_req_L.append(BRAM_cpt_req_scheme3_weightin_L[i])
                BRAM_BN_req_L.append(BRAM_BN_req_scheme3_L[i])
                BRAM_req_L.append(BRAM_req_scheme3_weightin_L[i])

            else:
                datain.append(datain_scheme3[i])
                dataout.append(dataout_scheme3[i])
                dataweight.append(dataweight_scheme3[i])

                BRAM_cpt_req_L.append(BRAM_cpt_req_scheme3_L[i])
                BRAM_BN_req_L.append(BRAM_BN_req_scheme3_L[i])
                BRAM_req_L.append(BRAM_req_scheme3_L[i])

        # logger.info('scheme_chosen=',scheme_chosen)
        # logger.info('scheme2_psumout=',scheme2_psumout)
        # logger.info('scheme3_weightin=',scheme3_weightin)
        return datain, dataout, dataweight

    def runtime(self):
        t_L = []
        pro_time = 0
        cycle_L = []

        datain, dataout, dataweight = self.compute_scheme_chosen()

        for i in range(0, self.LN):
            #### computing time
            if self.layer_type[i] == 0:  # conv
                if self.scheme_chosen[i] == 1:
                    t_L_c = (
                        math.ceil(self.N_in[i] / self.PE_num)
                        * self.N_out[i]
                        * (self.S[i] ** 2)
                        * self.K[i] ** 2
                        / (self.PE_size**2)
                        / self.f
                    )
                else:
                    t_L_c = (
                        math.ceil(self.N_out[i] / self.PE_num)
                        * self.N_in[i]
                        * (self.S[i] ** 2)
                        * self.K[i] ** 2
                        / (self.PE_size**2)
                        / self.f
                    )
            elif self.layer_type[i] == 2:  # FC
                t_L_c = self.N_in[i] * self.N_out[i] / self.datain_port_num / self.f

            else:  # pooling
                # t_L.append(N_in[i]*N_out[i]*((R[i]/2)**2)*K[i]**2/(convolver_num*PE_num*K[i]**2/convolver_2D_size**2)/f)
                t_L_c = (
                    math.ceil(self.N_out[i] / self.PE_num)
                    * self.N_in[i]
                    * ((self.R[i] / 2) ** 2)
                    * self.K[i] ** 2
                    / (self.PE_size**2)
                    / self.f
                )

            #### IO time
            if self.BN[i] == 1:
                IO = (
                    (datain[i] + dataout[i] + dataweight[i])
                    + (self.S[i] * 4 / self.minibatch)
                ) * self.bitwidth
            else:
                IO = (datain[i] + dataout[i] + dataweight[i]) * self.bitwidth
            t_L_d = IO / (self.BW_total * 1e9)

            #### t_L = max(t_compute, t_IO)
            t_L.append(max(t_L_c, t_L_d))
            # pdb.set_trace()

            cycle_L.append(t_L[i] * self.f)
            pro_time = pro_time + t_L[i]

        cycle_number = pro_time * self.f
        t_L_ms = [i * 1000 for i in t_L]

        # logger.info('t_L(ms)=',t_L_ms)
        # logger.info('Processing time per image(ms)=',pro_time*1000)

        # logger.info('cycle_L',cycle_L)
        # logger.info('cycle=',cycle_number)
        return pro_time, t_L

    def power(self):
        # mem
        Energy_DDR = 0
        Energy_BRAM = 0
        # cpt
        Energy_COMP = 0
        Energy_BN = 0

        Energy_DDR_L = []
        Energy_BRAM_L = []
        Energy_COMP_L = []
        Energy_BN_L = []

        convolver_2D_size = 5

        pro_time, t_L = self.runtime()
        datain, dataout, dataweight = self.compute_scheme_chosen()

        for i in range(0, self.LN):
            # mem DDR: kernel & input_images & out_images
            Energy_DDR_L.append((datain[i] + dataout[i] + dataweight[i]) * self.Emem_D)

            # mem BRAM: Intermediate data
            Energy_BRAM_L.append(
                self.N_in[i] * self.S[i] * self.R[i] * self.K[i] * self.Emem_B
                + self.S[i] * self.S[i] * self.N_out[i] * self.Emem_B * self.S[i]
            )

            # computation
            Energy_COMP_L.append(
                self.S[i] ** 2
                * self.N_in[i]
                * self.K[i] ** 2
                * self.N_out[i]
                * self.Ecpt
            )

            # BN
            if self.BN[i] == 1:
                Energy_BN_L.append(
                    self.N_out[i] / self.minibatch * 2 * self.Emem_D
                    + self.N_out[i] * self.S[i] ** 2 * self.Ecpt
                )
            else:
                Energy_BN_L.append(0)

            Energy_DDR = Energy_DDR + Energy_DDR_L[i]
            Energy_BRAM = Energy_BRAM + Energy_BRAM_L[i]
            Energy_COMP = Energy_COMP + Energy_COMP_L[i]
            Energy_BN = Energy_BN + Energy_BN_L[i]

        Energy_input_n = self.R[0] * self.R[0] * self.N_in[0] * self.Emem_D
        Energy_sta = self.P_sta * pro_time
        Energy = Energy_DDR + Energy_BRAM + Energy_COMP + Energy_BN + Energy_sta

        # logger.info('Energy_DDR=',Energy_DDR)
        # logger.info('Energy_BRAM=',Energy_BRAM)
        # logger.info('Energy_COMP=',Energy_COMP)
        # logger.info('Energy_sta=',Energy_sta)
        # logger.info('pro_time', pro_time)

        # logger.info('Energy per image(J)=',Energy)

        Power = Energy / pro_time
        # logger.info('Power(W)=',Power)

        return Power

    def bandwidth(self):
        Bandwidth_req_L = []
        Bandwidth_cpt_req_L = []
        Bandwidth_BN_req_L = []

        pro_time, t_L = self.runtime()
        datain, dataout, dataweight = self.compute_scheme_chosen()

        for i in range(0, self.LN):

            Bandwidth_cpt_req_L.append(
                (datain[i] + dataout[i] + dataweight[i]) * self.bitwidth / t_L[i] / 1e9
            )

            if self.BN[i] == 1:
                Bandwidth_BN_req_L.append(
                    self.S[i] * 4 / self.minibatch * self.bitwidth / t_L[i] / 1e9
                )
            else:
                Bandwidth_BN_req_L.append(0)

            Bandwidth_req_L.append(Bandwidth_cpt_req_L[i] + Bandwidth_BN_req_L[i])

        Bandwidth_req = max(Bandwidth_req_L)

        # logger.info('Bandwidth_req(Gb/s)=',Bandwidth_req)
        # logger.info('datain=',datain)
        # logger.info('dataout=',dataout)
        # logger.info('dataweight=',dataweight)

        # logger.info('Bandwidth_req_L=',Bandwidth_req_L)
        # logger.info('Bandwidth_BN_req_L=',Bandwidth_BN_req_L)
        # logger.info('Bandwidth_BN_cpt_L=',Bandwidth_cpt_req_L)
        return Bandwidth_req

    def DSP(self):
        DSP_BN = 20
        DSP = self.PE_size**2 * self.PE_num + DSP_BN
        # logger.info('DSP=',DSP)
        return DSP

    def Gops(self):
        op = 0
        op_L = []
        op_BN_L = []

        pro_time, t_L = self.runtime()

        for i in range(0, self.LN):

            op_L.append(self.S[i] ** 2 * self.N_in[i] * self.K[i] ** 2 * self.N_out[i])

            if self.BN[i] == 1:
                op_BN_L.append(self.N_out[i] * self.S[i] * self.S[i])
            else:
                op_BN_L.append(0)

            op = math.ceil(op + op_L[i] + op_BN_L[i])

        Gop = op * 2 / 1e9
        Gops = op * 2 / 1e9 / pro_time
        # logger.info('Gop=',Gop)
        # logger.info('Gop/s=',Gops)

        return Gops

    def energy(self):
        # mem
        Energy_DDR = 0
        Energy_BRAM = 0
        # cpt
        Energy_COMP = 0
        Energy_BN = 0

        Energy_DDR_L = []
        Energy_BRAM_L = []
        Energy_COMP_L = []
        Energy_BN_L = []

        convolver_2D_size = 5

        pro_time, t_L = self.runtime()
        datain, dataout, dataweight = self.compute_scheme_chosen()

        for i in range(0, self.LN):
            # mem DDR: kernel & input_images & out_images
            Energy_DDR_L.append((datain[i] + dataout[i] + dataweight[i]) * self.Emem_D)

            # mem BRAM: Intermediate data
            Energy_BRAM_L.append(
                self.N_in[i] * self.S[i] * self.R[i] * self.K[i] * self.Emem_B
                + self.S[i] * self.S[i] * self.N_out[i] * self.Emem_B * self.S[i]
            )

            # computation
            Energy_COMP_L.append(
                self.S[i] ** 2
                * self.N_in[i]
                * self.K[i] ** 2
                * self.N_out[i]
                * self.Ecpt
            )

            # BN
            if self.BN[i] == 1:
                Energy_BN_L.append(
                    self.N_out[i] / self.minibatch * 2 * self.Emem_D
                    + self.N_out[i] * self.S[i] ** 2 * self.Ecpt
                )
            else:
                Energy_BN_L.append(0)

            Energy_DDR = Energy_DDR + Energy_DDR_L[i]
            Energy_BRAM = Energy_BRAM + Energy_BRAM_L[i]
            Energy_COMP = Energy_COMP + Energy_COMP_L[i]
            Energy_BN = Energy_BN + Energy_BN_L[i]

        Energy_input_n = self.R[0] * self.R[0] * self.N_in[0] * self.Emem_D
        Energy_sta = self.P_sta * pro_time
        Energy = Energy_DDR + Energy_BRAM + Energy_COMP + Energy_BN + Energy_sta

        # logger.info('Energy_DDR=',Energy_DDR)
        # logger.info('Energy_BRAM=',Energy_BRAM)
        # logger.info('Energy_COMP=',Energy_COMP)
        # logger.info('Energy per image(J)=',Energy)

        return Energy
