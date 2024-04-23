import torch
import os
from subprocess import Popen

from crldse.env.eval import get_eval_metrics

path = '/app/m5out/stats.txt'


def extract_gem5_output_to_xml():
    """
    Convert the results of gem5 to XML format and extract the output file stats.txt of gem5
    """
    print("-----------------------START DEVORE-----------------------")
    f1 = open("/app/m5out/stats.txt")
    ss = "---------- Begin Simulation Statistics ----------"
    sr = f1.read().split(ss)
    f1.close()
    for i in range(len(sr)):
        f = open("/app/m5out/no%d.txt" % i, "w")
        f.write(sr[i] if i == 0 else ss + sr[i])
        f.close()
    print("-----------------------END DEVORE-------------------------")
    
def run_gem5_to_mcpat(state_dic, config):
    paths = config.get("paths", {})

    gem5_out = paths.get("gem5_out")
    mcpat_log = paths.get("mcpat_log")
    mcpat_script = paths.get("mcpat_script")
    gem5_script = paths.get("gem5_script")
    kernel_binary = paths.get("kernel_binary")
    disk_image = paths.get("disk_image")
    mcpat_binary = paths.get("mcpat_binary")
    gem5_binary = paths.get("gem5_binary")
    fs_script = paths.get("fs_script")

    if not os.path.exists(gem5_out):
        print("File '{}' not found.".format(gem5_out))
        return None

    try:
        print("-----------------------START GEM5 To McPAT-----------------------")
        process2 = Popen([
            "python3",
            gem5_script,
            gem5_out,
            "/m5out/config.json",
            f"/app/cMcPAT/mcpat/ProcessorDescriptionFiles/x86_AtomicSimpleCPU_template_core_{state_dic['core']}.xml",
            "-o",
            mcpat_script
        ])
        process2.wait()
        print("-----------------------END GEM5 To McPAT-----------------------")

        print("-----------------------START McPAT-----------------------")
        with open(mcpat_log, "w") as file_output:
            process3 = Popen([
                mcpat_binary,
                "-infile",
                mcpat_script,
                "-print_level",
                "5"
            ], stdout=file_output)
            process3.wait()
        print("-----------------------END McPAT-----------------------")

        print("-----------------------START print ENERGY-----------------------")
        eval_log_path = mcpat_log
        metrics = get_eval_metrics(eval_log_path, gem5_out)
        print("-----------------------END print ENERGY-----------------------")

        os.remove(gem5_out)
        os.remove(eval_log_path)
        os.remove(mcpat_script)

        return metrics
    except Exception as e:
        print("An error occurred:", str(e))
        cleanup(paths)
        return None

def cleanup(paths):
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)
    
if __name__=='__main__':
    run_gem5_to_mcpat()
    