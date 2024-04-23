import os
import time
import yaml
from subprocess import Popen

from crldse.env.eval import get_eval_metrics


def run_gem5_sim(obs, sim_config):
    cmd = "./gem5/build/X86/gem5.fast -re ./gem5/configs/deprecated/example/fs.py \
        --script=./parsec-image/benchmark_src/canneal_{}c_simdev.rcS -F 5000000000 \
        --cpu-type=TimingSimpleCPU --num-cpus={} --sys-clock={}GHz --caches --l2cache \
        --l1d_size={}kB --l1i_size={}kB --l2_size={}kB --l1d_assoc={} --l1i_assoc={} \
        --l2_assoc={} --kernel=./parsec-image/system/binaries/x86_64-vmlinux-2.6.28.4-smp \
        --disk-image=./parsec-image/system/disks/x86root-parsec.img".format(
            obs["core"],
            obs["core"],
            obs["sys_clock"],
            obs["l1d_size"],
            obs["l1i_size"],
            obs["l2_size"],
            obs["l1d_assoc"],
            obs["l1i_assoc"],
            obs["l2_assoc"]
        )
    print(cmd)
    process = Popen(cmd, shell=True)
    process.wait()
    return_code = process.returncode
    if return_code == 0:
        print("gem sim successfuly")
    return
        
            
def split_stats_file():
    flag = "---------- Begin Simulation Statistics ----------"
    with open("/app/m5out/stats.txt") as f:
        contents = f.read().split(flag)
    last_content = contents[-2]

    with open(f"/app/gem_sim_out/gem_output.txt", "w") as f:
        f.write(flag+last_content)


def run_mcpat(obs):
    trans = Popen(["python3", f"/app/cMcPAT/Scripts/GEM5ToMcPAT.py", \
        f"/app/gem_sim_out/gem_output.txt", f"/app/m5out/config.json", 
        f"/app/cMcPAT/mcpat/ProcessorDescriptionFiles/x86_AtomicSimpleCPU_template_core_{obs['core']}.xml", 
        "-o", f"/app/mcpat_in/test.xml"])
    trans.wait()

    with open("/app/mcpat_out/test.log", 'w') as file_output:
        proc = Popen([f"/app/cMcPAT/mcpat/mcpat", "-infile", f"/app/mcpat_in/test.xml", "-print_level", "5"], stdout=file_output)
        proc.wait()
    if proc.returncode == 0:
        print("mcpat sim successfuly")
    metrics = get_eval_metrics(f"/app/mcpat_out/test.log", f"/app/gem_sim_out/gem_output.txt")
    return metrics

def clean_up():
    for path in [f"/app/gem_sim_out/gem_output.txt", f"/app/mcpat_in/test.xml", f"/app/mcpat_out/test.log"]:
        if os.path.exists(path):
            os.remove(path)

def eval(obs):
    with open('/app/icrl/crldse/env/sim_config.yaml', "r") as f:
        sim_conf = yaml.safe_load(f)

    start_gem5_sim_time = time.time()
    print(f"============  start Gem5 simulator =============")
    run_gem5_sim(obs, sim_conf)
    print(f"============== end Gem5 simulator ==============")
    end_gem5_sim_time = time.time()
    print("gem sim cost time:", end_gem5_sim_time-start_gem5_sim_time)

    split_stats_file()

    start_mcpat_sim_time = time.time()
    print(f"================== start McPAT ==================")
    metrics = run_mcpat(obs)
    print(f"===================  end McPAT ===================")
    end_mcpat_sim_time = time.time()
    print("mcpat sim cost time:", end_mcpat_sim_time-start_mcpat_sim_time)

    if metrics:
        print(f"Evaluation successfully.")
        print(f"latency: {metrics['latency']} sec , Area: {metrics['Area']} mm^2 , energy: {metrics['energy']} mJ , power: {metrics['power']} W")
    else:
        print(f"Evaluation failed.")
    
    return metrics

if __name__=="__main__":
    default_obs = {
        "core": 4,
        "l1i_size": 256,
        "l1d_size": 256,
        "l2_size": 256,
        "l1d_assoc": 4,
        "l1i_assoc": 4,
        "l2_assoc": 4,
        "sys_clock": 2.8,
    }
    clean_up()
    res = eval(default_obs)
    print(res)