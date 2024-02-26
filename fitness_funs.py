# encoding: utf-8
import numpy as np
from gem5_mcpat_evaluation_2 import evaluation
from config import my_test_config

config = my_test_config()
# nnmodel = config.nnmodel
# layer_num = config.layer_num
# has_memory = True
# target = config.target
# DSP_THRESHOLD = config.DSP_THRESHOLD

# BW_THRESHOLD = config.BW_THRESHOLD
# BRAM_THRESHOLD = config.BRAM_THRESHOLD
POWER_THRESHOLD = 16
AREA_THRESHOLD = 165
LATENCY_THRESHOLD = 0.001


# 为了便于图示观察，试验测试函数为二维输入、二维输出
# 适应值函数：实际使用时请根据具体应用背景自定义
def fitness_(in_):
    DIM = 8
    status = dict()

    status["core"] = int(in_[0])
    status["l1i_size"] = int(in_[1])
    status["l1d_size"] = int(in_[2])
    status["l2_size"] = int(in_[3])
    status["l1d_assoc"] = int(in_[4])
    status["l1i_assoc"] = int(in_[5])
    status["l2_assoc"] = int(in_[6])
    status["sys_clock"] = int(in_[7]) / 10

    metrics = evaluation(status)

    if metrics != None:

        energy = metrics["energy"]
        area = metrics["Area"]
        runtime = metrics["latency"]
        power = metrics["power"]

    else:
        runtime = 10000
        power = 10000
        area = 10000

    # e_DSP = constraint_DSP(DSP)
    # e_energy = constraint_energy(energy)
    # e_BW = constraint_BW(BW)
    # e_BRAM = constraint_BRAM(BRAM)
    e_POWER = constraint_POWER(power)
    e_LAtency = constraint_LATENCY(runtime)
    e_AREA = constraint_AREA(area)
    if e_POWER > 0 or e_LAtency > 0 or e_AREA > 0:
        constrain_signal = False
    else:
        constrain_signal = True
    L1, L2, L3 = calc_Lj(e_POWER, e_LAtency, e_AREA)

    fit_1 = runtime + L1 * e_POWER + L2 * e_LAtency + L3 * e_AREA

    fit_2 = power + L1 * e_POWER + L2 * e_LAtency + L3 * e_AREA
    print(fit_2, fit_1)
    # degree_45 = ((in_[0]-in_[1]+in_[2])**2/2)**0.5
    # degree_135 = ((in_[0]+in_[1]+in_[2])**2/2)**0.5
    # fit_1 = 1-np.exp(-(degree_45)**2/0.5)*np.exp(-(degree_135-np.sqrt(200))**2/250)
    # fit_2 = 1-np.exp(-(degree_45)**2/5)*np.exp(-(degree_135)**2/350)
    return [fit_2, fit_1], constrain_signal


# def constraint_DSP(X):
#     return(max(0,X-DSP_THRESHOLD))
# def constraint_energy(X):
#     return (max(0, X-POWER_THRESHOLD))
# def constraint_BW(X):
#     return (max(0, X-BW_THRESHOLD))
# def constraint_BRAM(X):
#     return (max(0,X-BRAM_THRESHOLD))
def constraint_POWER(X):
    return max(0, X - POWER_THRESHOLD)


def constraint_AREA(X):
    return max(0, X - AREA_THRESHOLD)


def constraint_LATENCY(X):
    return max(0, X - LATENCY_THRESHOLD)


def calc_e1(X):
    """计算第一个约束的惩罚项"""
    e = X[0] + X[1] - 6
    return max(0, e)


def calc_e2(X):
    """计算第二个约束的惩罚项"""
    e = 3 * X[0] - 2 * X[1] - 5
    return max(0, e)


def calc_Lj(e1, e2, e3):
    """根据每个粒子的约束惩罚项计算Lj权重值，e1, e2列向量，表示每个粒子的第1个第2个约束的惩罚项值"""
    # 注意防止分母为零的情况
    if (e1 + e2 + e3) <= 0:
        return 0, 0, 0
    else:
        L1 = e1 / (e1 + e2 + e3)
        L2 = e2 / (e1 + e2 + e3)
        L3 = e2 / (e1 + e2 + e3)
    return L1, L2, L3
