import os
from subprocess import Popen

from crldse.env.eval import get_eval_metrics
from crldse.utils.core import read_config


def run_gem5_simulation(state_dic):
    command_template = """
        /parsec-tests2/gem5_2/gem5/build/X86/gem5.fast -re \
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
        --disk-image=/parsec-tests2/parsec-image/system/disks/x86root-parsec.img
    """
    
    command = command_template.format(
        state_dic["core"],
        state_dic["core"],
        state_dic["sys_clock"],
        state_dic["l1d_size"],
        state_dic["l1i_size"],
        state_dic["l2_size"],
        state_dic["l1d_assoc"],
        state_dic["l1i_assoc"],
        state_dic["l2_assoc"]
    )

    # exection
    os.system(command)


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
            f"/parsec-tests2/cmcpat/cMcPAT/mcpat/ProcessorDescriptionFiles/x86_AtomicSimpleCPU_template_core_{state_dic['core']}.xml",
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

if __name__=="__main__":
    config = read_config("sim_config.yaml")
    state_dic = {
        "core": 3,
        "sys_clock": 2,
        "l1d_size": 256,
        "l1i_size": 256,
        "l2_size": 64,
        "l1d_assoc": 8,
        "l1i_assoc": 8,
        "l2_assoc": 8,
    }
    run_gem5_to_mcpat(state_dic, config)