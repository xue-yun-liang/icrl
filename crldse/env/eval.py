import os
import re
import time
import subprocess as subp
from optparse import OptionParser
from subprocess import Popen

import numpy as np
from crldse.utils.core import read_config

mcpat_bin = "mcpat"


class parse_node:
    def __init__(this, key=None, value=None, indent=0):
        this.key = key
        this.value = value
        this.indent = indent
        this.leaves = []

    def append(this, n):
        this.leaves.append(n)

    def get_tree(this, indent):
        padding = ' ' * indent * 2
        me = padding + this.__str__()
        kids = list(map(lambda x: x.get_tree(indent + 1), this.leaves))
        return me + '\n' + ''.join(kids)

    def getValue(this, key_list):
        if (this.key == key_list[0]):
            if len(key_list) == 1:
                return this.value
            else:
                kids = list(map(lambda x: x.getValue(key_list[1:]), this.leaves))
                return ''.join(kids)
        return ''

    def __str__(this):
        return 'k: ' + str(this.key) + ' v: ' + str(this.value)


class parser:

    def dprint(this, astr):
        if this.debug:
            print (this.name, astr)

    def __init__(this, data_in):
        this.debug = False
        this.name = 'mcpat:mcpat_parse'

        buf = open(data_in)

        this.root = parse_node('root', None, -1)
        trunk = [this.root]

        for line in buf:

            indent = len(line) - len(line.lstrip())
            equal = '=' in line
            colon = ':' in line
            useless = not equal and not colon
            items = list(map(lambda x: x.strip(), line.split('=')))

            branch = trunk[-1]

            if useless:
                pass

            elif equal:
                assert (len(items) > 1)

                n = parse_node(key=items[0], value=items[1], indent=indent)
                branch.append(n)

                this.dprint('new parse_node: ' + str(n))

            else:

                while (indent <= branch.indent):
                    this.dprint('poping branch: i: ' + str(indent) + \
                                ' r: ' + str(branch.indent))
                    trunk.pop()
                    branch = trunk[-1]

                this.dprint('adding new leaf to ' + str(branch))
                n = parse_node(key=items[0], value=None, indent=indent)
                branch.append(n)
                trunk.append(n)

    def get_tree(this):
        return this.root.get_tree(0)

    def getValue(this, key_list):
        value = this.root.getValue(['root'] + key_list)
        assert (value != '')
        return value


# runs McPAT and gives you the total energy in mJs
def getevaluation(index_1_mcpat,index_2_gem5):
    energy,runtime,Aera,power= getEnergy(index_1_mcpat, index_2_gem5)

    metrics = {
                'latency':runtime,  # unit: sec
                'Area':Aera,
                'energy':energy,    # unit: mJ 功耗
                'power':power       # 性能
                
            }


    print ("energy is %f mJ" % energy)
    return metrics


def getEnergy(mcpatoutputFile, statsFile):
    leakage, dynamic,Aera= readMcPAT(mcpatoutputFile)
    runtime = getTimefromStats(statsFile)
    energy = (leakage + dynamic) * runtime
    print ("leakage: %f W, dynamic: %f W ,Aera: %f mm^2 and runtime: %f sec" % (leakage, dynamic,Aera,runtime))

    return energy * 1000,runtime,Aera,(leakage+dynamic)


def readMcPAT(mcpatoutputFile):
    print ("Reading simulation time from: %s" % mcpatoutputFile)
    p = parser(mcpatoutputFile)

    leakage = p.getValue(['Processor:', 'Total Leakage'])
    dynamic = p.getValue(['Processor:', 'Runtime Dynamic'])
    Aera = p.getValue(['Processor:', 'Area'])
    leakage = re.sub(' W', '', leakage)
    dynamic = re.sub(' W', '', dynamic)
    Aera = re.sub('m', '', Aera)
    Aera = Aera[:-4]
    return (float(leakage), float(dynamic),float(Aera))


def getTimefromStats(statsFile):
    print ("Reading simulation time from: %s" % statsFile)
    F = open(statsFile)
    ignores = re.compile(r'^---|^$')
    statLine = re.compile(r'([a-zA-Z0-9_\.:+-]+)\s+([-+]?[0-9]+\.[0-9]+|[0-9]+|nan)')
    retVal = None
    for line in F:
        # ignore empty lines and lines starting with "---"
        if not ignores.match(line):
            statKind = statLine.match(line).group(1)
            statValue = statLine.match(line).group(2)
            if statKind == 'simSeconds':
                retVal = float(statValue)
                break  # no need to parse the whole file once the requested value has been found
    F.close()
    return retVal



def run_gem5_sim(obs, sim_config):
    print(obs)
    """sim step1: pass the design parameters of the state to gem5"""
    # if the cache size is too small, mcpat will return error
    # so, need to give a new lower bound or if we get error, give a done for env
    cmd = "/app/gem5/build/X86/gem5.fast -re /app/gem5/configs/deprecated/example/fs.py \
        --script=/app/parsec-image/benchmark_src/canneal_{}c_simdev.rcS -F 5000000000 \
        --cpu-type=TimingSimpleCPU --num-cpus={} --sys-clock={}GHz --caches --l2cache \
        --l1d_size={}kB --l1i_size={}kB --l2_size={}kB --l1d_assoc={} --l1i_assoc={} \
        --l2_assoc={} --kernel=/app/parsec-image/system/binaries/x86_64-vmlinux-2.6.28.4-smp \
        --disk-image=/app/parsec-image/system/disks/x86root-parsec.img".format(
        obs["core"], obs["core"], obs["sys_clock"], obs["l1d_size"], obs["l1i_size"],
        obs["l2_size"], obs["l1d_assoc"], obs["l1i_assoc"],obs["l2_assoc"])
    print(cmd)
    proc = Popen(cmd, shell=True)
    proc.wait()
    return_code = proc.returncode
    if return_code == 0:
        print("gem sim successfuly")
    else:
        print("gem sim fail")
    return
        

def split_stats_file():
    """sim step2: spilt the stats.txt(out file) of gem5 by flag, and get the 4th file as out file"""
    flag = "---------- Begin Simulation Statistics ----------"
    with open("./m5out/stats.txt") as f:
        contents = f.read().split(flag)
    last_content = contents[-2]

    with open(f"/app/gem_sim_out/gem_output.txt", "w") as f:
        f.write(flag+last_content)


def run_mcpat(obs):
    """sim step3: convert the results of GEM5 to XML format and run mcpat sim, return metrics"""
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
        metrics = getevaluation(f"/app/mcpat_out/test.log", f"/app/gem_sim_out/gem_output.txt")
    else:
        print("mcpat sim fail")
        metrics = {"latency":1, "Area":1, "power":1, "energy":1}
    
    return metrics


def clean_up():
    for path in [f"/app/gem_sim_out/gem_output.txt", f"/app/mcpat_in/test.xml", f"/app/mcpat_out/test.log"]:
        if os.path.exists(path):
            os.remove(path)


def sim_func(obs, sim_conf):
    clean_up()
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


class evaluation_function:
    """
    A class that actually executes gem5 and mcpat, and returns the evaluation result metric
    """

    def __init__(self, obs) -> None:
        self.obs = obs
        self.eval_len = len(obs)
        self.config_data = None
        self.metrics = None
    

    def eval(self, obs_in):
        assert len(obs_in) == self.eval_len
        for key, value in zip(self.obs.keys(), obs_in):
            self.obs[key] = value
        self.metrics = sim_func(self.obs, self.config_data)
        return self.metrics

    def print_eval_res(self) -> None:
        print("obs is:", self.obs)
        print("eval res:", self.metrics)
        
         

if __name__=='__main__':
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
    
    eval = evaluation_function(default_obs)
    eval.print_eval_res()
    obs = [4,256,256,512,2,2,2,2.8]
    eval.eval(obs)
    eval.print_eval_res()

    