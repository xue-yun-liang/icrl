import numpy as np
import yaml

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def read_config(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)