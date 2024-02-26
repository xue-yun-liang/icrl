# encoding: utf-8
import numpy as np
import xlwt
from Mopso import *
from config import my_test_config

config = my_test_config()
# nnmodel = config.nnmodel
# layer_num = config.layer_num
has_memory = True


#### step2 assign platform
# constrain
# target = config.target
# DSP_THRESHOLD = config.DSP_THRESHOLD
# POWER_THRESHOLD = config.POWER_THRESHOLD
# BW_THRESHOLD = config.BW_THRESHOLD
# BRAM_THRESHOLD = config.BRAM_THRESHOLD
def main():
    w = 0.8  # 惯性因子
    c1 = 0.1  # 局部速度因子
    c2 = 0.1  # 全局速度因子
    particals = 50  # 粒子群的数量
    cycle_ = 10  # 迭代次数
    mesh_div = 10  # 网格等分数量
    thresh = 300  # 外部存档阀值
    min_ = np.array([1, 1, 1, 6, 1, 1, 1, 20])  # 粒子坐标的最小值
    max_ = np.array([16, 12, 12, 16, 4, 4, 4, 40])  # 粒子坐标的最大值
    # for index in range(layer_num):
    #     min_ = np.append(min_,1)
    #     max_ = np.append(max_,3)

    mopso_ = Mopso(particals, w, c1, c2, max_, min_, thresh, mesh_div)  # 粒子群实例化
    pareto_in, pareto_fitness = mopso_.done(
        cycle_
    )  # 经过cycle_轮迭代后，pareto边界粒子
    runtimelist = mopso_.runtime_list
    powerlist = mopso_.power_list
    pareto_fitness.tolist()
    workbook = xlwt.Workbook(encoding="ascii")
    worksheet = workbook.add_sheet("1")
    worksheet.write(0, 0, "period")
    worksheet.write(0, 1, "power")
    worksheet.write(0, 2, "latancy")
    best_runtime_record = 100000
    for index, best_runtime in enumerate(pareto_fitness):
        worksheet.write(index + 1, 0, index + 1)
        worksheet.write(index + 1, 1, best_runtime[0])
        worksheet.write(index + 1, 2, best_runtime[1])
    name = "./img_txt/" + "_" + "mopso" + "_" + ".xls"
    workbook.save(name)

    workbook = xlwt.Workbook(encoding="ascii")
    worksheet = workbook.add_sheet("1")
    worksheet.write(0, 0, "period")
    worksheet.write(0, 1, "latancy")
    worksheet.write(0, 2, "power")

    for index, best_runtime in enumerate(runtimelist):
        worksheet.write(index + 1, 0, index + 1)
        worksheet.write(index + 1, 1, best_runtime)
        worksheet.write(index + 1, 2, powerlist[index])

    name = "./img_txt/" + "_" + "mopso" + "_" + "allvalue" + ".xls"
    workbook.save(name)

    np.savetxt("./img_txt/pareto_in.txt", pareto_in)  # 保存pareto边界粒子的坐标
    np.savetxt(
        "./img_txt/pareto_fitness.txt", pareto_fitness
    )  # 打印pareto边界粒子的适应值
    print("\n", "pareto边界的坐标保存于：/img_txt/pareto_in.txt")
    print("pareto边界的适应值保存于：/img_txt/pareto_fitness.txt")
    print("\n,迭代结束,over")


if __name__ == "__main__":
    main()
