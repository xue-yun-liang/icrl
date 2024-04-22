import random
import math

import numpy as np
import torch

from crldse.utils import core


class actor_random():
	def make_policy(self, design_space, dimension_index):
		'''
		return the index of dimension's sample_box 
		rather than the actual value for convinience
		'''
		return random.randint(0,design_space.dimension_box[dimension_index].get_scale()-1)

class actor_e_greedy():
	def __init__(self):
		self.greedy_possiblity = 0.7
		self.sample_range = int(2)
		
	def action_choose(self, qfunction, design_space, dimension_index, ratio):
		# constrain the sample range
		# python range is [a,b), so the up bound requre a +1
		current_index = design_space.get_dimension_current_index(dimension_index)
		# get the sample range
		sample_bottom = max(0, current_index - self.sample_range)
		sample_top = min(int(design_space.get_dimension_scale(dimension_index)), current_index + self.sample_range + 1)
		
		if(random.random() < self.greedy_possiblity):
		# greedy search best action
			self.best_action_index = 0
			self.best_qvalue = 0
			# find the best action in that dimension
			for action_index in range(int((design_space.get_dimension_scale(dimension_index)))):
			#for action_index in range(sample_bottom, sample_top):
				status = design_space.sample_one_dimension(dimension_index, action_index)
				with torch.no_grad():
					#### compute the q value
					step = (dimension_index+1) / design_space.get_lenth()
					step = torch.tensor(step).float().view(1)
					status = core.status_normalize(status, design_space)
					variable = core.status_to_Variable(status)
					variable = torch.cat((variable, step), dim = -1)

					qvalue = qfunction(variable)
				# compare and find the best q value
				if(qvalue > self.best_qvalue):
					self.best_action_index = action_index
					self.best_qvalue = qvalue
		else:
		# random choose an action
			self.best_action_index = random.randint(0, design_space.get_dimension_scale(dimension_index) - 1)
			# self.best_action_index = random.randint(sample_bottom, sample_top - 1)
		return self.best_action_index

	'''
	def ratiochange_action_choose(self, qfunction, design_space, dimension_index, ratio):
		#### constrain the sample range
		#### python range is [a,b), so the up bound requre a +1
		current_index = design_space.get_dimension_current_index(dimension_index)
		sample_bottom = max(0, current_index - self.sample_range)
		sample_top = min(int(design_space.get_dimension_scale(dimension_index)), current_index + self.sample_range + 1)
		
		if(random.random() < self.greedy_possiblity**ratio):
		#### greedy search best action
			self.best_action_index = 0
			self.best_qvalue = 0
			#### find the best action in that dimension
			for action_index in range(int((design_space.get_dimension_scale(dimension_index)))):
			#for action_index in range(sample_bottom, sample_top):
				status = design_space.sample_one_dimension(dimension_index, action_index)
				with torch.no_grad():
					#### compute the q value
					step = dimension_index / design_space.get_lenth()
					step = torch.tensor(step).float().view(1)
					status = status_normalize(status, design_space)
					variable = status_to_Variable(status)
					variable = torch.cat((variable, step), dim = -1)

					qvalue = qfunction(variable)
				##### compare and find the best q value
				if(qvalue > self.best_qvalue):
					self.best_action_index = action_index
					self.best_qvalue = qvalue
		else:
		#### random choose an action
			self.best_action_index = random.randint(0, design_space.get_dimension_scale(dimension_index) - 1)
			#self.best_action_index = random.randint(sample_bottom, sample_top - 1)
		return self.best_action_index
	'''
	
	def random_action_choose(self, qfunction, design_space, dimension_index, ratio):
		# constrain the sample range
		# python range is [a,b), so the up bound requre a +1
		current_index = design_space.get_dimension_current_index(dimension_index)
		sample_bottom = max(0, current_index - self.sample_range)
		sample_top = min(int(design_space.get_dimension_scale(dimension_index)), current_index + self.sample_range + 1)		

		#### random choose an action
		self.best_action_index = random.randint(0, design_space.get_dimension_scale(dimension_index) - 1)
		#self.best_action_index = random.randint(sample_bottom, sample_top - 1)
		return self.best_action_index

	def best_action_choose(self, qfunction, design_space, dimension_index):
		##### greedy search best action
		self.true_best_action_index = 0
		self.true_best_qvalue = 0
		##### find the best action in that dimension
		for action_index in range(int((design_space.get_dimension_scale(dimension_index)))):
			status = design_space.sample_one_dimension(dimension_index, action_index)
			with torch.no_grad():
				step = (dimension_index+1) / design_space.get_lenth()
				step = torch.tensor(step).float().view(1)
				status = core.status_normalize(status, design_space)
				variable = core.status_to_Variable(status)
				variable = torch.cat((variable, step), dim = -1)
				
				qvalue = qfunction(variable)
			##### compare and find the best q value
			if(qvalue > self.true_best_qvalue):
				self.true_best_action_index = action_index
				self.true_best_qvalue = qvalue
		return self.true_best_action_index

class actor_policyfunction():
	def action_choose(self, policyfunction, design_space, dimension_index):
		status = design_space.get_status()
		status_normalization = core.status_normalize(status, design_space)
		probs = policyfunction(core.status_to_Variable(status_normalization), dimension_index)
		use_noise = False
		if(use_noise):		
			noise = torch.normal(mean = torch.zeros_like(probs), std = 0.005)
			probs_noise = probs + noise
			probs_noise = torch.clamp(probs_noise, 0, 1)
			action_index = probs_noise.multinomial(num_samples = 1).data
		else:
			action_index = probs.multinomial(num_samples = 1).data
		#### compute entropy
		entropy = -(probs * probs.log()).sum()
		#### use multinomial to realize the sampling of policy function
		action_index_tensor = core.index_to_one_hot(len(probs), action_index)
		#### use onehot index to restore compute graph
		prob_sampled = (probs * action_index_tensor).sum()
		log_prob_sampled = prob_sampled.log()

		#if(dimension_index == 8):
		#	logger.info(f"\nstep:{dimension_index}, probs:{probs}")
		return entropy, action_index, log_prob_sampled

	def action_choose_with_no_grad(self, policyfunction, design_space, dimension_index, std = 0.1, is_train = True):
		status = design_space.get_status()
		with torch.no_grad():
			status_normalization = core.status_normalize(status, design_space)
			probs = policyfunction(core.status_to_Variable(status_normalization), dimension_index)

			if(is_train):
				model = design_space.get_dimension_model(dimension_index)
				if(model["name"] == "normal"):
					noise = torch.normal(mean = torch.zeros_like(probs), std = std)
					probs_noise = probs + noise
					probs_noise = torch.clamp(probs_noise, 0, 1)
					#probs_noise = abs(probs_noise)
					#probs_noise = probs_noise/probs_noise.sum()
					#logger.info(f"probs_noise:{probs_noise}")
					#probs_noise = probs
				elif(model["name"] == "one_hot"):
					noise = torch.normal(mean = torch.zeros_like(probs), std = model["param"])
					probs_noise = probs + noise
					probs_noise = torch.clamp(probs_noise, 0, 1)
					#probs_noise = abs(probs_noise)
					#probs_noise = probs_noise/probs_noise.sum()
			else:
				probs_noise = probs

			'''
			if(dimension_index == 2):
				logger.info(f"probs:{probs}")
				logger.info(f"noise:{noise}")
				#pdb.set_trace()
			'''
		

			#pdb.set_trace()
			#probs_noise = torch.abs(probs + noise)
			#probs_noise = torch.nn.functional.softmax(probs + noise)
			#logger.info(f"original:{probs}")
			#logger.info(f"noise:{probs_noise}")
			#### use multinomial to realize the sampling of policy function
			action_index = probs_noise.multinomial(num_samples = 1).data

		return action_index, probs_noise[action_index].data

	def action_choose_with_no_grad_2(self, policyfunction,policyfunction_2,design_space, dimension_index,w1,w2, std=0.01,is_train=True):
		status = design_space.get_status()
		with torch.no_grad():
			status_normalization = core.status_normalize(status, design_space)
			probs1 = policyfunction(core.status_to_Variable(status_normalization), dimension_index)
			probs2 = policyfunction_2(core.status_to_Variable(status_normalization), dimension_index)

			probs = w1*probs1+w2*probs2
			if (is_train):
				model = design_space.get_dimension_model(dimension_index)
				if (model["name"] == "normal"):
					noise = torch.normal(mean=torch.zeros_like(probs), std=std)
					probs_noise = probs + noise
					probs_noise = torch.clamp(probs_noise, 0, 1)
				# probs_noise = abs(probs_noise)
				# probs_noise = probs_noise/probs_noise.sum()
				# logger.info(f"probs_noise:{probs_noise}")
				# probs_noise = probs
				elif (model["name"] == "one_hot"):
					noise = torch.normal(mean=torch.zeros_like(probs), std=model["param"])
					probs_noise = probs + noise
					probs_noise = torch.clamp(probs_noise, 0, 1)
			# probs_noise = abs(probs_noise)
			# probs_noise = probs_noise/probs_noise.sum()
			else:
				probs_noise = probs

			'''
			if(dimension_index == 2):
				logger.info(f"probs:{probs}")
				logger.info(f"noise:{noise}")
				#pdb.set_trace()
			'''

			# pdb.set_trace()
			# probs_noise = torch.abs(probs + noise)
			# probs_noise = torch.nn.functional.softmax(probs + noise)
			# logger.info(f"original:{probs}")
			# logger.info(f"noise:{probs_noise}")
			#### use multinomial to realize the sampling of policy function
			action_index = probs_noise.multinomial(num_samples=1).data

		return action_index, probs_noise[action_index].data

	def action_choose_with_no_grad_3(self, policyfunction,policyfunction_2,policyfunction_3,design_space, dimension_index,signol, std=0.1,is_train=True):
		status = design_space.get_status()
		with torch.no_grad():
			status_normalization = core.status_normalize(status, design_space)
			probs1 = policyfunction(core.status_to_Variable(status_normalization), dimension_index)
			probs2 = policyfunction_2(core.status_to_Variable(status_normalization), dimension_index)
			probs3 = policyfunction_3(core.status_to_Variable(status_normalization), dimension_index)

			if signol==1:
				probs = probs1
				#logger.info("1")
			elif signol==2:
				probs = probs2
				#logger.info("2")
			else:
				probs = probs3
				#logger.info("3")

			if (is_train):
				model = design_space.get_dimension_model(dimension_index)
				if (model["name"] == "normal"):
					noise = torch.normal(mean=torch.zeros_like(probs), std=std)
					probs_noise = probs + noise
					probs_noise = torch.clamp(probs_noise, 0, 1)
				# probs_noise = abs(probs_noise)
				# probs_noise = probs_noise/probs_noise.sum()
				# logger.info(f"probs_noise:{probs_noise}")
				# probs_noise = probs
				elif (model["name"] == "one_hot"):
					noise = torch.normal(mean=torch.zeros_like(probs), std=model["param"])
					probs_noise = probs + noise
					probs_noise = torch.clamp(probs_noise, 0, 1)
			# probs_noise = abs(probs_noise)
			# probs_noise = probs_noise/probs_noise.sum()
			else:
				probs_noise = probs

			'''
			if(dimension_index == 2):
				logger.info(f"probs:{probs}")
				logger.info(f"noise:{noise}")
				#pdb.set_trace()
			'''

			# pdb.set_trace()
			# probs_noise = torch.abs(probs + noise)
			# probs_noise = torch.nn.functional.softmax(probs + noise)
			#### use multinomial to realize the sampling of policy function
			action_index = probs_noise.multinomial(num_samples=1).data

		return action_index, probs_noise[action_index].data


	def action_choose_DDPG(self, policyfunction, design_space, dimension_index):
		with torch.no_grad():
			status = design_space.get_status()
			status = core.status_normalize(status, design_space)
			probs = policyfunction(core.status_to_Variable(status), dimension_index)
			probs_softmax = torch.softmax(probs, dim = -1)
			#### use multinomial to realize the sampling of policy function
			action_index = probs_softmax.multinomial(num_samples = 1).data
			#action_index = torch.argmax(probs)
		return action_index, probs