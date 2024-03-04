from subprocess import Popen, PIPE
from getevaluation import getevaluation
from logger import Logger
import time
import os
import math
import logging
import subprocess


def evaluation(status):
    # init logger
    logger = Logger.get_logger()
    
    # just accpet a param "stats":dict()
    # return the evaluation metris:dict()

    # process by gem5 simulation , simulation result trans to mcpat
    # mcpat trans to final metric
    core = str(status["core"])
    benchmarksize = ""
    l1i_size = str(int(math.pow(2, int(status["l1i_size"]))))
    l1d_size = str(int(math.pow(2, int(status["l1d_size"]))))
    l2_size = str(int(math.pow(2, int(status["l2_size"]))))
    l1d_assoc = str(int(math.pow(2, status["l1d_assoc"])))
    l1i_assoc = str(int(math.pow(2, status["l1i_assoc"])))
    l2_assoc = str(int(math.pow(2, status["l2_assoc"])))
    sys_clock = str(status["sys_clock"])

    logger.info("core = %s", core)
    logger.info("l1i_size = %s", l1i_size)
    logger.info("l1d_size = %s", l1d_size)
    logger.info("l2_size = %s", l2_size)
    logger.info("l1d_assoc = %s", l1d_assoc)
    logger.info("l1i_assoc = %s", l1i_assoc)
    logger.info("l2_assoc = %s", l2_assoc)
    logger.info("sys_clock = %s", sys_clock)

    # core = "3"
    # benchmarksize =""
    # l1i_size ="256"
    # l1d_size ="256"
    # l2_size="64"
    # l1d_assoc="8"
    # l1i_assoc="8"
    # l2_assoc="8"
    # sys_clock="2"

    start = time.time()
    bar = "========================="

    # Execute the simulation command
    logger.info(bar, "START SIMULATION", bar)
    os.system(
        "/parsec-tests2/gem5_2/gem5/build/X86/gem5.fast -re \
            /parsec-tests2/gem5_2/gem5/configs/example/fs.py \
            --script=/parsec-tests2/parsec-image/benchmark_src/canneal_{}c_simdev.rcS \
            -F 5000000000  --cpu-type=TimingSimpleCPU --num-cpus={} \
            --sys-clock='{}GHz' \
            --caches --l2cache   \
            --l1d_size='{}kB' \
            --l1i_size='{}kB' \
            --l2_size='{}kB' \
            --l1d_assoc={} \
            --l1i_assoc={} \
            --l2_assoc={} \
            --kernel=/parsec-tests2/parsec-image/system/binaries/x86_64-vmlinux-2.6.28.4-smp \
            --disk-image=/parsec-tests2/parsec-image/system/disks/x86root-parsec.img".format(
            core,
            core,
            sys_clock,
            l1d_size,
            l1i_size,
            l2_size,
            l1d_assoc,
            l1i_assoc,
            l2_assoc,
        )
    )
    logger.info(bar, "END SIMULATER", bar)

    # extract the output file stats.txt of gem5
    logger.info(bar + "START DEVORE", bar)
    f1 = open("/m5out/stats.txt")
    ss = "---------- Begin Simulation Statistics ----------"
    sr = f1.read().split(ss)
    f1.close()
    for i in range(len(sr)):
        f = open("/m5out/%d.txt" % i, "w")
        f.write(sr[i] if i == 0 else ss + sr[i])
        f.close()
    logger.info(bar, "END DEVORE", bar)

    try:
        if os.path.exists("/m5out/3.txt"):
            logger.info(bar, "START GEM5 To McPAT", bar)
            command_2 = [
                "python3",
                "/parsec-tests2/cmcpat/cMcPAT/Scripts/GEM5ToMcPAT.py",
                "/m5out/3.txt",
                "/m5out/config.json",
                "/parsec-tests2/cmcpat/cMcPAT/mcpat/ProcessorDescriptionFiles/x86_AtomicSimpleCPU_template_core_{}.xml".format(
                    core
                ),
                "-o",
                "/parsec-tests2/cmcpat/cMcPAT/Scripts/test.xml",
            ]
            process2 = Popen(command_2)
            process2.wait()
            logger.info(bar, "END GEM5 To McPAT", bar)

            logger.info(bar, "START McPAT", bar)
            file_output = open(
                "/parsec-tests2/cmcpat/cMcPAT/mcpatresult/test2.log", "w"
            )
            command_3 = [
                "/parsec-tests2/cmcpat/cMcPAT/mcpat/mcpat",
                "-infile",
                "/parsec-tests2/cmcpat/cMcPAT/Scripts/test.xml",
                "-logger.info_level",
                "5",
            ]
            process3 = Popen(command_3, stdout=file_output)
            process3.wait()
            logger.info(bar, "END McPAT", bar)

            logger.info(bar, "START logger.info ENERGY", bar)
            # command_4 = ["python3",
            # "/parsec-tests2/cmcpat/cMcPAT/Scripts/logger.info_energy.py" ,
            # "/parsec-tests2/cmcpat/cMcPAT/mcpatresult/test2.log",
            # "/parsec-tests2/gem5_2/gem5/m5out/3.txt"]
            # process4 = Popen(command_4)
            # process4.wait()
            metrics = getevaluation(
                "/parsec-tests2/cmcpat/cMcPAT/mcpatresult/test2.log", "/m5out/3.txt"
            )
            logger.info(bar, "END logger.info ENERGY", bar)

            os.remove("/m5out/3.txt") if os.path.exists("/m5out/3.txt") else None
            (
                os.remove("/parsec-tests2/cmcpat/cMcPAT/mcpatresult/test2.log")
                if os.path.exists("/parsec-tests2/cmcpat/cMcPAT/mcpatresult/test2.log")
                else None
            )
            (
                os.remove("/parsec-tests2/cmcpat/cMcPAT/Scripts/test.xml")
                if os.path.exists("/parsec-tests2/cmcpat/cMcPAT/Scripts/test.xml")
                else None
            )
            end = time.time()
            logger.info("The process_1 running time:{}".format(end - start))
            return metrics
        else:
            return None
    except:
        os.remove("/m5out/3.txt") if os.path.exists("/m5out/3.txt") else None
        (
            os.remove("/parsec-tests2/cmcpat/cMcPAT/mcpatresult/test2.log")
            if os.path.exists("/parsec-tests2/cmcpat/cMcPAT/mcpatresult/test2.log")
            else None
        )
        (
            os.remove("/parsec-tests2/cmcpat/cMcPAT/Scripts/test.xml")
            if os.path.exists("/parsec-tests2/cmcpat/cMcPAT/Scripts/test.xml")
            else None
        )
        logger.info(f"current status can't be evaluated")
        return None
