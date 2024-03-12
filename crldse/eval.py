import sys
import re
import os
import json
import types
import math
import subprocess as subp
from optparse import OptionParser

from crldse.logger import Logger

logger = Logger.get_logger()
mcpat_bin = "mcpat"


class parse_node:
    def __init__(this, key=None, value=None, indent=0):
        this.key = key
        this.value = value
        this.indent = indent
        this.leaves = []

    def __str__(this):
        return "key: " + str(this.key) + " value: " + str(this.value)

    def append(this, n):
        this.leaves.append(n)

    def tree(this, indent):
        padding = " " * indent * 2
        me = padding + this.__str__()
        kids = list(map(lambda x: x.get_tree(indent + 1), this.leaves))
        return me + "\n" + "".join(kids)

    def value(this, key_list):
        if this.key == key_list[0]:
            if len(key_list) == 1:
                return this.value
            else:
                kids = list(map(lambda x: x.get_value(key_list[1:]), this.leaves))
                return "".join(kids)
        return ""


class parser:
    def __init__(this, file):
        this.debug = False
        this.name = "mcpat:mcpat_parse"

        buf = open(file)

        this.root = parse_node("root", None, -1)
        trunk = [this.root]

        for line in buf:
            indent = len(line) - len(line.lstrip())
            equal = "=" in line
            colon = ":" in line
            useless = not equal and not colon
            items = list(map(lambda x: x.strip(), line.split("=")))
            branch = trunk[-1]

            if useless:
                pass
            elif equal:
                assert len(items) > 1
                n = parse_node(key=items[0], value=items[1], indent=indent)
                branch.append(n)
                this.dlogger.info("new parse_node: " + str(n))
            else:
                while indent <= branch.indent:
                    this.print(
                        "poping branch: i: " + str(indent) + " r: " + str(branch.indent)
                    )
                    trunk.pop()
                    branch = trunk[-1]
                this.print("adding new leaf to " + str(branch))
                n = parse_node(key=items[0], value=None, indent=indent)
                branch.append(n)
                trunk.append(n)

    def print(this, astr):
        """print the file's name"""
        if this.debug:
            print(this.name, astr)

    def get_tree(this):
        return this.root.get_tree(0)

    def get_value(this, key_list):
        value = this.root.getValue(["root"] + key_list)
        assert value != ""
        return value


def read_mcpat(mcpatOutputFile):
    """parse the mcpat output file"""
    print("Reading simulation time from: %s" % mcpatOutputFile)
    p = parser(mcpatOutputFile)

    leakage = p.get_value(["Processor:", "Total Leakage"])
    dynamic = p.get_value(["Processor:", "Runtime Dynamic"])
    Aera = p.get_value(["Processor:", "Area"])
    leakage = re.sub(" W", "", leakage)
    dynamic = re.sub(" W", "", dynamic)
    Aera = re.sub("m", "", Aera)
    Aera = Aera[:-4]
    return (float(leakage), float(dynamic), float(Aera))


def get_evaluation(index_1_mcpat, index_2_gem5):
    """runs McPAT, return the total energy in mJs"""
    energy, runtime, Aera, power = get_energy(index_1_mcpat, index_2_gem5)

    metrics = {
        "latency": runtime,  # unit: sec
        "Area": Aera,
        "energy": energy,  # unit: mJ
        "power": power,
    }

    logger.info("energy is %f mJ" % energy)
    return metrics


def get_energy(mcpatoutputFile, statsFile):
    leakage, dynamic, Aera = read_mcpat(mcpatoutputFile)
    runtime = get_time_from_stats(statsFile)
    energy = (leakage + dynamic) * runtime
    print(
        "leakage: %f W, dynamic: %f W ,Aera: %f mm^2 and runtime: %f sec"
        % (leakage, dynamic, Aera, runtime)
    )

    return energy * 1000, runtime, Aera, (leakage + dynamic)


def get_time_from_stats(stats_file):
    print("Reading simulation time from: %s" % stats_file)
    F = open(stats_file)
    ignores = re.compile(r"^---|^$")
    stat_line = re.compile(r"([a-zA-Z0-9_\.:+-]+)\s+([-+]?[0-9]+\.[0-9]+|[0-9]+|nan)")
    ret_val = None
    for line in F:
        # ignore empty lines and lines starting with "---"
        if not ignores.match(line):
            stat_kind = stat_line.match(line).group(1)
            stat_value = stat_line.match(line).group(2)
            if stat_kind == "simSeconds":
                ret_val = float(stat_value)
                break  # no need to parse the whole file once the requested value has been found
    F.close()
    return ret_val
