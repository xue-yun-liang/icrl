import yaml
import math
import torch
import numpy as np

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def read_config(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)
    
def state_normalize(state, design_space):
	state_copy = dict()
	for item in design_space.dimension_box:
		name = item.get_name()
		state_copy[name] = state[name] / item.get_range_upbound()
	return state_copy

def action_value_normalize(action_list, design_space):
	action_list_copy = list()
	for index, item in enumerate(design_space.dimension_box):
		action_list_copy.append(action_list[index] / len(item.sample_box))
	return action_list_copy

def compute_action_distance(action_list_a, action_list_b, design_space):
	action_list_a_normalized = action_value_normalize(action_list_a, design_space)
	action_list_b_normalized = action_value_normalize(action_list_b, design_space)
	distance = 0
	for i, j in zip(action_list_a_normalized, action_list_b_normalized):
		distance = distance + (i-j)**2
	return distance

def action_normalize(action_tensor, design_space, step):
	action_tensor = action_tensor / (design_space.get_dimension_scale(step)-1)

def state_to_list(state):
	_list = []
	for index in state:
		_list.append(state[index])
	return _list

def state_to_torch_tensor(state):
	_list = state_to_list(state)
	_ndarray = np.array(_list)
	_tensor = torch.from_numpy(_ndarray)
	return _tensor

def state_to_Variable(state):
	"""
	Convert 'state' to torch.sensor,then wrap the tensor into a PyTorch 
 	Variable object, which is typically used to represent nodes 
  	in a computational graph and allows for automatic differentiation
	"""
	_tensor = state_to_torch_tensor(state)
	_Variable = torch.autograd.Variable(_tensor).float()
	return _Variable	

def index_to_one_hot(scale, action_index):
	_tensor = torch.zeros(scale)
	_tensor.scatter_(dim = 0, index = action_index, value = 1)
	return _tensor

def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

def normal_density(x, mean, sigma):
	return 1/((2 * 3.1415)**0.5 * sigma) \
		   * math.exp(- (x - mean)**2/(2 * sigma**2))

def get_normal_tensor(design_space, action_index, dimension_index, model_sigma):
	sample_box = design_space.get_dimension_sample_box(dimension_index)
	normal_list = []
	for sample_index, value in enumerate(sample_box):
		normal_list.append(normal_density(sample_index, action_index, model_sigma))
	normal_tensor = torch.from_numpy(np.array(normal_list))
	normal_tensor = normal_tensor / normal_tensor.sum()
	return normal_tensor

def get_log_prob(policyfunction, design_space, state, action_index, dimension_index):
	state_normalization = state_normalize(state, design_space)
	probs = policyfunction(state_to_Variable(state_normalization), dimension_index)
	#### compute entropy
	entropy = -(probs * probs.log()).sum()
	#### use multinomial to realize the sampling of policy function
	action_index_tensor = index_to_one_hot(len(probs), action_index)
	#### use onehot index to restore compute graph
	prob_sampled = (probs * action_index_tensor).sum()
	log_prob_sampled = prob_sampled.log()


	return entropy, log_prob_sampled

def get_kldivloss_and_log_prob(policyfunction, design_space, state, action_index, dimension_index):
	state_normalization = state_normalize(state, design_space)
	probs = policyfunction(state_to_Variable(state_normalization), dimension_index)
	#### compute entropy
	entropy = -(probs * probs.log()).sum()
	#### compute kl div between probs and target distribution
	model = design_space.get_dimension_model(dimension_index)
	if(model["name"] == "normal"):
		target_distribution = get_normal_tensor(design_space, action_index, dimension_index, model["param"]).float()
		if(dimension_index == 2): print(f"normal: target_distribution{target_distribution}")	
	elif(model["name"] == "one_hot"):
		target_distribution = index_to_one_hot(len(probs), action_index)
	kldivloss = torch.nn.functional.kl_div(probs.log(), target_distribution, reduction = "sum")
	#### use multinomial to realize the sampling of policy function
	action_index_tensor = index_to_one_hot(len(probs), action_index)
	#### use onehot index to restore compute graph
	prob_sampled = (probs * action_index_tensor).sum()
	log_prob_sampled = prob_sampled.log()


	return entropy, kldivloss, log_prob_sampled

def sample_index_from_2d_array(array):
    sampled_indices = []
    for sub_array in array:
        probabilities = torch.tensor(sub_array, dtype=torch.float)
        probabilities /= torch.sum(probabilities)  # 将概率归一化为和为1
        sampled_index = torch.multinomial(probabilities, 1).item()  # 使用multinomial进行抽样
        sampled_indices.append(sampled_index)
    return sampled_indices