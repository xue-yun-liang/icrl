import numpy as np
import geatpy as ea
from evaluation import evaluation_function
from config import my_test_config
import random
from gem5_mcpat_evaluation_3 import evaluation
from matplotlib import pyplot as plt

from multiprocessing import Pool
import xlwt
import pdb

# global parameters, NIND is the number of population
NIND = 50

#### step1 assign model
config = my_test_config()
# nnmodel = config.nnmodel
# layer_num = config.layer_num
# has_memory = True

#### step2 assign platform
# constrain
# target = config.target
AREA_THRESHOLD = config.AREA_THRESHOLD


#### step3 assign goal
config.config_check()

record = 1
my_period = 0
best_runtime_now = 10000
runtime_list = list()
best_power_now = 10000
power_list = list()


class DSE(ea.Problem):
    def __init__(self):
        name = "DSE"

        M = 2  # the dimension of object function
        maxormins = [1, 1]  # 1 means to minmize the ob function

        """
		d1:PE_num d2:PE_size d3:f d4:minibatch d5:bitwidth
		"""
        Dim = 8
        varTypes = [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]  # 1 for discrete value, 0 for continues value
        lb = [1, 1, 1, 6, 1, 1, 1, 20]  # low bound
        ub = [16, 12, 12, 16, 4, 4, 4, 40]  # up bound

        lbin = [
            1
        ] * Dim  # 1 means include the low bound, 0 means do not include the low bound
        ubin = [1] * Dim

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        global my_period, best_runtime_now, best_power_now, power_list, runtime_list
        vars = pop.Phen  # Phen is the variables matrix

        vec_core = vars[:, [0]]
        vec_l1i_size = vars[:, [1]]
        vec_l1d_size = vars[:, [2]]
        vec_l2_size = vars[:, [3]]
        vec_l1d_assoc = vars[:, [4]]
        vec_l1i_assoc = vars[:, [5]]
        vec_l2_assoc = vars[:, [6]]
        vec_sys_clock = vars[:, [7]]

        vec_runtime = np.zeros(NIND).reshape(-1, 1)
        vec_energy = np.zeros(NIND).reshape(-1, 1)
        vec_power = np.zeros(NIND).reshape(-1, 1)
        vec_area = np.zeros(NIND).reshape(-1, 1)

        bar = "====================="
        for index in range(NIND):
            print(bar, str(my_period + 1), bar)
            my_period = my_period + 1
            status = dict()
            status["core"] = int(vec_core[index])
            status["l1i_size"] = int(vec_l1i_size[index])
            status["l1d_size"] = int(vec_l1d_size[index])
            status["l2_size"] = int(vec_l2_size[index])
            status["l1d_assoc"] = int(vec_l1d_assoc[index])
            status["l1i_assoc"] = int(vec_l1i_assoc[index])
            status["l2_assoc"] = int(vec_l2_assoc[index])
            status["sys_clock"] = int(vec_sys_clock[index]) / 10

            metrics = evaluation(status)

            if metrics != None:

                energy = metrics["energy"]
                area = metrics["Area"]
                runtime = metrics["latency"]
                power = metrics["power"]

            else:
                runtime = 10000
                power = 10000
                energy = 10000
                area = 10000

            if area < AREA_THRESHOLD:
                best_runtime_now = runtime
                best_power_now = power
            else:
                best_runtime_now = 10000
                best_power_now = 10000

            runtime_list.append(best_runtime_now)
            power_list.append(best_power_now)

            vec_runtime[index] = runtime
            vec_energy[index] = energy
            vec_power[index] = power
            vec_area[index] = area - AREA_THRESHOLD

        pop.ObjV = np.hstack([vec_runtime, vec_power])
        # elif(goal == "energy"):
        # 	pop.ObjV = vec_energy
        # elif(goal == "latency&energy"):
        # 	pop.ObjV = vec_re
        pop.CV = np.hstack([vec_area])

    def calReferObjV(self):
        referenceObjV = np.array([50])
        return referenceObjV


def run(iindex):
    print(f"%%%%%%%%%%%%%%%TEST{iindex} START%%%%%%%%%%%%%")
    global my_period, best_runtime_now, best_power_now, power_list, runtime_list

    seed = iindex * 10000
    atype = int(iindex / 10)

    np.random.seed(seed)
    random.seed(seed)

    problem = DSE()
    Encoding = "RI"
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    # myalgorithm = ea.soea_DE_rand_1_bin_templet(problem, population)
    # myalgorithm = ea.soea_DE_best_1_bin_templet(problem, population)
    myalgorithm = ea.moea_NSGA2_templet(problem, population)
    myalgorithm.MAXGEN = 10
    # myalgorithm.mutOper.F = 0.5
    # myalgorithm.recOper.XOVR = 0.7
    myalgorithm.drawing = 1

    print("________________algorithm run_________________")
    referenceObjV = np.array([50])
    NDSet = myalgorithm.run()
    print(NDSet)
    print(NDSet.ObjV)

    objectvalue1 = list()
    objectvalue2 = list()
    for item in NDSet.ObjV:
        objectvalue1.append(item[0])
        objectvalue2.append(item[1])
    print(objectvalue1)
    print(objectvalue2)
    # plt.figure(figsize=(9, 6))
    # plt.xlabel('latency')
    # plt.ylabel('power')
    # plt.scatter(objectvalue1, objectvalue2, c='red', s=25, alpha=0.4, marker='o', label="third")
    # plt.show()

    # population.save()

    # print("obj_trace", obj_trace[:, 1])
    # print("var_trace", var_trace)

    workbook = xlwt.Workbook(encoding="ascii")
    worksheet = workbook.add_sheet("1")
    worksheet.write(0, 0, "period")
    worksheet.write(0, 1, "latancy")
    worksheet.write(0, 2, "power")
    best_runtime_record = 100000
    for index, best_runtime in enumerate(objectvalue1):
        worksheet.write(index + 1, 0, index + 1)
        worksheet.write(index + 1, 1, best_runtime)
        worksheet.write(index + 1, 2, objectvalue2[index])
    name = "record/runtime/" + "_" + "NSGA" + "_" + str(iindex) + ".xls"
    workbook.save(name)

    print(f"**************TEST{iindex} END***********")

    workbook = xlwt.Workbook(encoding="ascii")
    worksheet = workbook.add_sheet("1")
    worksheet.write(0, 0, "period")
    worksheet.write(0, 1, "latancy")
    worksheet.write(0, 2, "power")
    for index, best_runtime in enumerate(runtime_list):
        worksheet.write(index + 1, 0, index + 1)
        worksheet.write(index + 1, 1, best_runtime)
        worksheet.write(index + 1, 2, power_list[index])
    name = "record/runtime/" + "_" + "NSGA" + "_" + str(iindex) + "allvalue" + ".xls"
    workbook.save(name)

    # print(f"**************TEST{iindex} END***********")

    # workbook = xlwt.Workbook(encoding = 'ascii')
    # worksheet = workbook.add_sheet("1")
    # worksheet.write(0, 0, "period")
    # worksheet.write(0, 1, "power")
    # for index, best_power in enumerate(power_list):
    # 	worksheet.write(index+1, 0, index+1)
    # 	worksheet.write(index+1, 1, best_power)
    # name = "record/runtime/" + "_" + "NSGA" + "_" + str(iindex) +"power"+".xls"
    # workbook.save(name)

    # print(f"**************TEST{iindex} END***********")


if __name__ == "__main__":
    USE_MULTIPROCESS = False
    TEST_BOUND = 1

    if USE_MULTIPROCESS:
        iindex_list = list()
        for i in range(TEST_BOUND):
            iindex_list.append(i)

        pool = Pool(3)
        pool.map(run, iindex_list)
        pool.close()
        pool.join()
    else:
        for iindex in range(TEST_BOUND):
            run(iindex)
