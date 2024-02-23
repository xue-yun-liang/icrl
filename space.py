import torch
import numpy
import random
import pdb

import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt

from evaluation import evaluation_function

class dimension_discrete():
	def __init__(self, name, default_value, step, rrange, frozen = False, model = {"name":"normal", "param":0.4}):
		'''
		"name"-string
		"default_value"-int
		"step"-int
		"rrange"-[low-int,high-int]
		'''
		self.name = name
		self.default_value = default_value
		self.current_value = default_value
		self.step = step
		self.frozen = frozen
		self.model = model

		assert(rrange[0] <= rrange[-1])
		self.rrange = rrange
		self.scale = (self.rrange[-1] - self.rrange[0])//self.step + 1

		self.default_index = int((default_value - self.rrange[0])//self.step)
		self.current_index = self.default_index
		'''
		sample_box offers the sample space for discrete dimension
		every item in sample_box is a avaiable value for that dimension
		NOW: thr range of sample_box is [rrange[0] ~ rrange[0]+step*(scale-1) rather than [rrange[0] ~ rrange[-1]]
		that's not a big deal
		'''
		self.sample_box = []
		for idx in range(int(self.scale)):
			self.sample_box.append(self.rrange[0]+idx*step)

	def set(self, sample_index):
		assert (sample_index >= 0 and sample_index <=self.scale-1)
		self.current_index = sample_index
		self.current_value = self.sample_box[sample_index]

	def original_set(self, sample_index):
		assert (sample_index >= 0 and sample_index <=self.scale-1)
		self.default_index = sample_index
		self.default_value = self.sample_box[sample_index]

	def reset(self):
		self.current_index = self.default_index
		self.current_value = self.default_value

	def get_name(self):
		return self.name
	
	def get_scale(self):
		return self.scale
	
	def get_range_upbound(self):
		return self.rrange[-1]
	
	def sample(self, sample_index):
		assert (sample_index >= 0 and sample_index <=self.scale-1)
		if(self.frozen == False):
			self.current_index = sample_index
			self.current_value = self.sample_box[sample_index]
		return self.current_value
	
	def get_current_index(self):
		return self.current_index
	
	def get_current_value(self):
		return self.current_value
	
	def get_sample_box(self):
		return self.sample_box
	
	def froze(self):
		self.frozen = True

	def release(self):
		self.frozen = False

	def get_model(self):
		return self.model

	#### new function 2021/07/26
	def get_norm_current_value(self):
		return self.get_current_value()/self.get_range_upbound()

class design_space():
	def __init__(self):
		'''
		dimension_box is a list of dict which is consist of two item, "name":str and "dimension":dimension_discrete
		'''
		self.dimension_box = []
		#self.evaluation = evaluation_function(nnmodel = "VGG16", target = "normal")
		self.lenth = 0
		self.scale = 1

	def append(self,dimension_discrete):
		self.dimension_box.append(dimension_discrete)
		self.lenth = self.lenth + 1
		self.scale = self.scale * dimension_discrete.get_scale()

	def get_status(self):
		'''
		status is a dict class that can be used for matching of dimension "name":"dimension_value"
		'''
		status = dict()
		for item in self.dimension_box:
			status[item.get_name()] = item.get_current_value()
		return status
	
	def get_status_value(self):
		status_value = list()
		for item in self.dimension_box:
			status_value.append(item.get_current_value())
		return status_value
	
	def get_action_list(self):
		action_list = list()
		for item in self.dimension_box:
			action_list.append(item.get_current_index())
		return action_list
	
	def print_status(self):
		for item in self.dimension_box:
			print(item.get_name(),item.get_current_value())

	def sample_one_dimension(self, dimension_index, sample_index):
		assert (dimension_index >= 0 and dimension_index <= self.lenth-1)
		self.dimension_box[dimension_index].sample(sample_index)
		return self.get_status()
	
	def set_one_dimension(self, dimension_index, sample_index):
		assert (dimension_index >= 0 and dimension_index <= self.lenth-1)
		self.dimension_box[dimension_index].set(sample_index)
		return self.get_status()	
		
	def status_set(self, best_action_list):
		for dimension, action in zip(self.dimension_box, best_action_list):
			dimension.set(action)
		return self.get_status()
	
	def original_status_set(self, best_action_list):
		for dimension, action in zip(self.dimension_box, best_action_list):
			dimension.original_set(action)
		return self.get_status()
	
	def status_reset(self):
		for dimension in self.dimension_box:
			dimension.reset()
		return self.get_status()
	
	def get_lenth(self):
		return self.lenth
	
	def get_scale(self):
		return self.scale
	
	def get_dimension_current_index(self, dimension_index):
		return self.dimension_box[dimension_index].get_current_index()
	
	def get_dimension_scale(self, dimension_index):
		return self.dimension_box[dimension_index].get_scale()
	
	def get_dimension_sample_box(self, dimension_index):
		return self.dimension_box[dimension_index].sample_box
	
	def froze_one_dimension(self, dimension_index):
		self.dimension_box[dimension_index].froze()

	def release_one_dimension(self, dimension_index):
		self.dimension_box[dimension_index].release()

	def froze_dimension(self, dimension_index_list):
		for index in dimension_index_list:
			self.froze_one_dimension(index)

	def release_dimension(self, dimension_index_list):
		for index in dimension_index_list:
			self.release_one_dimension(index)
			
	def get_dimension_model(self, dimension_index):
		return self.dimension_box[dimension_index].get_model()
	
	#### new function, for ppo, require numpy and torch
	def get_obs(self):
		obs_list = list()
		for item in self.dimension_box:
			obs_list.append(item.get_norm_current_value())
		obs = numpy.array(obs_list)
		#obs = torch.from_numpy(obs)
		return obs


def create_space(layer_num):
	DSE_action_space = design_space()
	num_param = 0.5
	type_param = 0.1

	## initial DSE_action_space
	PE_x = dimension_discrete(
		name = "PE_x", 
		default_value = 2, 
		step = 2, 
		rrange = [2,30],
		model = {"name":"normal", "param":num_param}
		#model = {"name":"one_hot", "param":type_param}
	)
	PE_y = dimension_discrete(
		name = "PE_y", 
		default_value = 2, 
		step = 2,
		rrange = [2,30],
		model = {"name":"normal", "param":num_param}
		#model = {"name":"one_hot", "param":type_param}
	)
	f = dimension_discrete(
		name = "f",
		default_value = 1.5,
		step = 0.1, 
		rrange = [0.8,2.6],
		model = {"name":"normal", "param":num_param}
		#model = {"name":"one_hot", "param":type_param}
	)
	minibatch = dimension_discrete(
		name = "minibatch",
		default_value = 5,
		step = 5,
		rrange = [1,50],
		model = {"name":"normal", "param":num_param}
		#model = {"name":"one_hot", "param":type_param}
	)
	bitwidth = dimension_discrete(
		name = "bitwidth",
		default_value = 2,
		step = 1,
		rrange = [0,2],
		#model = {"name":"normal", "param":num_param}
		model = {"name":"one_hot", "param":type_param}
	)

	BW_frequency = dimension_discrete(
		name = "BW_frequency",
		default_value = 0,
		step = 1,
		rrange = [0,4],
		#model = {"name":"normal", "param":num_param}
		model = {"name":"one_hot", "param":type_param}
	)
	#self.BW_frequency_list = [1.333, 1.6, 1.866, 2.133, 2.4]
	

	BW_bitwidth = dimension_discrete(
		name = "BW_bitwidth",
		default_value = 64,
		step = 8,
		rrange = [32,128],
		model = {"name":"normal", "param":num_param}
		#model = {"name":"one_hot", "param":type_param}
	) 

	BW_channel = dimension_discrete(
		name = "BW_channel",
		default_value = 1,
		step = 1,
		rrange = [1,2],
		#model = {"name":"normal", "param":num_param}
		model = {"name":"one_hot", "param":type_param}
	)

	BRAM_total = dimension_discrete(
		name = "BRAM_total",
		default_value = 0,
		step = 1,
		rrange = [0,13],
		model = {"name":"normal", "param":num_param}
		#model = {"name":"one_hot", "param":type_param}
	)
	#self.BRAM_list = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

	scheme_chosen_list = list()
	for index in range(layer_num):
		scheme_chosen_layer = dimension_discrete(
			name = ("scheme_chosen" + str(index + 1)),
			default_value = 1,
			step = 1,
			rrange = [1,3],
			#model = {"name":"normal", "param":0.4}
			model = {"name":"one_hot", "param":type_param}
		)	
		scheme_chosen_list.append(scheme_chosen_layer)

	DSE_action_space = design_space()
	DSE_action_space.append(PE_x)
	DSE_action_space.append(PE_y)
	DSE_action_space.append(f)
	DSE_action_space.append(minibatch)
	DSE_action_space.append(bitwidth)
	DSE_action_space.append(BW_frequency)
	DSE_action_space.append(BW_bitwidth)
	DSE_action_space.append(BW_channel)
	DSE_action_space.append(BRAM_total)
	for item in scheme_chosen_list:
		DSE_action_space.append(item)	

	return DSE_action_space

##########lijianfei_gem5###################################
def create_space_gem5():
	DSE_action_space = design_space()
	num_param = 0.5
	type_param = 0.1

	core = dimension_discrete(
		name="core",
		default_value=1,
		step=1,
		rrange=[1, 16],
		model={"name": "normal", "param": num_param}
		# model = {"name":"one_hot", "param":type_param}
	)

	l1i_size = dimension_discrete(
		name="l1i_size",
		default_value=1,
		step=1,
		rrange=[1, 12],
		model={"name": "normal", "param": num_param}
		# model = {"name":"one_hot", "param":type_param}
	)
	l1d_size = dimension_discrete(
		name="l1d_size",
		default_value=1,
		step=1,
		rrange=[1, 12],
		model={"name": "normal", "param": num_param}
		# model = {"name":"one_hot", "param":type_param}
	)
	l2_size = dimension_discrete(
		name="l2_size",
		default_value=6,
		step=1,
		rrange=[6, 16],
		model={"name": "normal", "param": num_param}
		# model = {"name":"one_hot", "param":type_param}
	)
	l1d_assoc = dimension_discrete(
		name="l1d_assoc",
		default_value=1,
		step=1,
		rrange=[1, 4],
		model={"name": "normal", "param": num_param}
		# model = {"name":"one_hot", "param":type_param}
	)
	l1i_assoc = dimension_discrete(
		name="l1i_assoc",
		default_value=1,
		step=1,
		rrange=[1, 4],
		model={"name": "normal", "param": num_param}
		# model = {"name":"one_hot", "param":type_param}
	)

	l2_assoc = dimension_discrete(
		name="l2_assoc",
		default_value=1,
		step=1,
		rrange=[1, 4],
		model={"name": "normal", "param": num_param}
		# model = {"name":"one_hot", "param":type_param}
	)

	sys_clock = dimension_discrete(
		name="sys_clock",
		default_value=2,
		step=0.1,
		rrange=[2, 4],
		model={"name": "normal", "param": num_param}
		# model = {"name":"one_hot", "param":type_param}
	)


	DSE_action_space = design_space()
	DSE_action_space.append(core)
	DSE_action_space.append(l1i_size)
	DSE_action_space.append(l1d_size)
	DSE_action_space.append(l2_size)
	DSE_action_space.append(l1d_assoc)
	DSE_action_space.append(l1i_assoc)
	DSE_action_space.append(l2_assoc)
	DSE_action_space.append(sys_clock)

	return DSE_action_space




class environment():

	def __init__(self, layer_num, model, target, goal, constraints, nlabel, algo, test = False, delay_reward = True):
		self.layer_num = layer_num
		self.model = model
		self.target = target
		self.goal = goal
		self.constraints = constraints
		self.nlabel = nlabel
		self.algo = algo
		self.test = test
		self.delay_reward = delay_reward
				
		self.best_result = 1e10
		self.best_reward = 0
		self.sample_time = 0


		log_addr = "./record/objectvalue/"
		log_name = log_addr + self.model + "_" + self.target + "_" + self.goal + "_" + self.algo + "_" + str(self.nlabel)
		self.result_log = open(log_name, "w")
		print(f"Period\tResult", end = '\n', file = self.result_log)

		self.design_space = create_space(self.layer_num)

		self.design_space_dimension = self.design_space.get_lenth()
		self.action_dimension_list = list()
		self.action_limit_list = list()
		for dimension in self.design_space.dimension_box:
			self.action_dimension_list.append(int(dimension.get_scale()))
			self.action_limit_list.append(int(dimension.get_scale() - 1))
		
		self.evaluation = evaluation_function(self.model, self.target)
		

	def reset(self):
		self.design_space.status_reset()
		return self.design_space.get_obs()

	def step(self, step, act, deterministic=False):
		if(deterministic):
			if(torch.is_tensor(act)): act = torch.argmax(act, dim=-1).item()
			else: act = torch.argmax(torch.as_tensor(act).view(-1), dim=-1).item()
		else:
			if(self.algo == "SAC"):
				if(torch.is_tensor(act)): 
					act = torch.softmax(act, dim = -1)
					act = int(act.multinomial(num_samples = 1).data.item())
				if(isinstance(act, numpy.ndarray)):
					act = torch.as_tensor(act, dtype=torch.float32).view(-1)
					act = torch.softmax(act, dim = -1)
					act = int(act.multinomial(num_samples = 1).data.item())
			elif(self.algo == "PPO"):
				pass	

		current_status = self.design_space.get_status()
		next_status = self.design_space.sample_one_dimension(step, act)
		obs = self.design_space.get_obs()

		if(step < (self.design_space.get_lenth() - 1)):
			not_done = 1
		else:
			not_done = 0

		if(not_done):
			if(self.delay_reward):
				reward = float(0)
			else:
				self.evaluation.update_parameter(next_status, has_memory = True)
				runtime, t_L = self.evaluation.runtime()
				runtime = runtime * 1000
				power,DSP = self.evaluation.power(), self.evaluation.DSP()
				energy = self.evaluation.energy()
				BW = self.evaluation.BW_total
				BRAM = self.evaluation.BRAM_total

				self.constraints.update({"DSP":DSP, "POWER":power, "BW":BW, "BRAM":BRAM})

				if(self.goal == "latency"):
					reward = 1000/(runtime * self.constraints.get_punishment())
					result = runtime
				elif(self.goal == "energy"):
					reward = 1000/(energy * self.constraints.get_punishment())
					result = energy 
				elif(self.goal == "latency&energy"):
					reward = 1000/((runtime * energy) * self.constraints.get_punishment())
					result = runtime * energy 

				if((result < self.best_result) and self.constraints.is_all_meet()): self.best_result = result
				if((reward > self.best_reward) and self.constraints.is_all_meet()): self.best_reward = reward
				
				if(not self.test):
					self.sample_time += 1
					print(f"{self.sample_time}\t{self.best_result}", end = "\n", file = self.result_log)
		else:
			self.evaluation.update_parameter(next_status, has_memory = True)
			runtime, t_L = self.evaluation.runtime()
			runtime = runtime * 1000
			power,DSP = self.evaluation.power(), self.evaluation.DSP()
			energy = self.evaluation.energy()
			BW = self.evaluation.BW_total
			BRAM = self.evaluation.BRAM_total

			self.constraints.update({"DSP":DSP, "POWER":power, "BW":BW, "BRAM":BRAM})

			if(self.goal == "latency"):
				reward = 1000/(runtime * self.constraints.get_punishment())
				result = runtime
			elif(self.goal == "energy"):
				reward = 1000/(energy * self.constraints.get_punishment()) 
				result = energy
			elif(self.goal == "latency&energy"):
				reward = 1000/((runtime * energy) * self.constraints.get_punishment()) 
				result = runtime * energy

			if((result < self.best_result) and self.constraints.is_all_meet()): self.best_result = result
			if((reward > self.best_reward) and self.constraints.is_all_meet()): self.best_reward = reward

			if(not self.test):
				self.sample_time += 1
				print(f"{self.sample_time}\t{self.best_result}", end = "\n", file = self.result_log)

		done = not not_done

		return obs, reward, done, {}

	def sample(self, step):
		idx = random.randint(0, self.design_space.get_dimension_scale(step)-1)
		pi = torch.zeros(int(self.design_space.get_dimension_scale(step)))
		pi[idx] = 1
		return pi

def tsne3D(vector_list, reward_list, method):
	action_array = np.array(vector_list)
	reward_continue_array = np.array(reward_list)

	tsne = manifold.TSNE(n_components = 2, init = "pca", random_state = 501)
	print(f"Start to load t-SNE")
	x_tsne = tsne.fit_transform(action_array)

	x_min, x_max = x_tsne.min(0), x_tsne.max(0)
	x_norm = (x_tsne - x_min)/(x_max - x_min)
	#pdb.set_trace()

	fig_3D = plt.figure()
	tSNE_3D = plt.axes(projection = '3d')
	tSNE_3D.scatter3D(x_norm[:, 0], x_norm[:, 1], reward_continue_array, c = reward_continue_array, vmax = 20, cmap = "rainbow", alpha = 0.5)
	#tSNE_3D.scatter3D(x_norm[:, 0], x_norm[:, 1], reward_continue_array, c = reward_continue_array, cmap = "rainbow", alpha = 0.5)
	tSNE_3D.set_xlabel("x")
	tSNE_3D.set_ylabel("y")
	tSNE_3D.set_zlabel("Reward")
	tSNE_3D.set_zlim((0, 20))
	tSNE_3D.set_zticks([0,5,10,15,20])	
	fname = method + "_" + "tSEN_3D" + ".png"
	fig_3D.savefig(fname, format = "png")