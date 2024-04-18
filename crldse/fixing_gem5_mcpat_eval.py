import time
import os
import math
from subprocess import Popen

from crldse.eval import get_evaluation
from logger import logger

logger = logger.get_logger()
bar = "------------------------------"

# the eval process:
# step1: Pass the design parameters of the state to gem5
# step2: Convert the results of GEM5 to XML format
# step3: Pass the gem5 output file in XML format to MCPAT

class evaluation_function:
    """A class that actually executes gem5 and mcpat, and returns the evaluation result metric"""
    def __init__(self, status) -> None:
        self.status = status
        self.eval_args = None
        
    def get_args(self) -> None:
        core = str(self.status["core"])
        l1i_size = str(int(math.pow(2, int(self.status["l1i_size"]))))
        l1d_size = str(int(math.pow(2, int(self.status["l1d_size"]))))
        l2_size = str(int(math.pow(2, int(self.status["l2_size"]))))
        l1d_assoc = str(int(math.pow(2, self.status["l1d_assoc"])))
        l1i_assoc = str(int(math.pow(2, self.status["l1i_assoc"])))
        l2_assoc = str(int(math.pow(2, self.status["l2_assoc"])))
        sys_clock = str(self.status["sys_clock"])

        eval_args = {
            "core":core,
            "l1i_size": l1i_size,
            "l1d_size": l1d_size,
            "l2_size": l2_size,
            "l1d_assoc": l1d_assoc,
            "l1i_assoc":l1i_assoc,
            "l2_assoc":l2_assoc,
            "sys_clock":sys_clock
        }
        
        self.eval_args = eval_args
        return
        
        
    def print_args(self, args) -> None:
        logger.info("core = %s", args["core"])
        logger.info("l1i_size = %s", args["l1i_size"])
        logger.info("l1i_size = %s", args["l1i_size"])
        logger.info("l2_size = %s", args["l2_size"])
        logger.info("l1d_assoc = %s", args["l1d_assoc"])
        logger.info("l1i_assoc = %s", args["l1i_assoc"])
        logger.info("l2_assoc = %s", args["l2_assoc"])
        logger.info("sys_clock = %s", args["sys_clock"])
    
    # step1: Pass the design parameters of the state to gem5
    def sim_gem5(self):
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
            self.args["core"],
            self.args["core"],
            self.args["sys_clock"],
            self.args["l1d_assoc"],
            self.args["l1i_assoc"],
            self.args["l2_size"],
            self.eval_args["l1d_assoc"],
            self.eval_args["l1i_assoc"],
            self.eval_args["l2_assoc"],
        )
    )
    logger.info(bar, "END SIMULATER", bar)
    
    
    # step2: Convert the results of GEM5 to XML format
    # and extract the output file stats.txt of gem5
    def extract_gem5_output_to_xml(self):
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
    
    # step3: Pass the gem5 output file in XML format to MCPAT
    def sim_mcpat(self):  
        try:
            if os.path.exists("/m5out/3.txt"):
                print(bar, "START GEM5 To McPAT", bar)
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
                # the trule eval step
                eval_log_path = "/parsec-tests2/cmcpat/cMcPAT/mcpatresult/test2.log"
                metrics = get_evaluation(eval_log_path, "/m5out/3.txt")
                logger.info(bar, "END logger.info ENERGY", bar)

                os.remove("/m5out/3.txt") if os.path.exists("/m5out/3.txt") else None
                (os.remove(eval_log_path) if os.path.exists(eval_log_path) else None)
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

# a exmaple of eval_agrs:
# core = "3"
# l1i_size ="256"
# l1d_size ="256"
# l2_size="64"
# l1d_assoc="8"
# l1i_assoc="8"
# l2_assoc="8"
# sys_clock="2"
