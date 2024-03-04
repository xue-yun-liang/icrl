from space import dimension_discrete
from space import design_space
from space import create_space_gem5
from space import tsne3D
from actor import actor_policyfunction, get_log_prob
from evaluation import evaluation_function
from config_2 import my_test_config
from multiprocessing import Pool
from gem5_mcpat_evaluation_2 import evaluation
import torch
import random
import numpy
import pdb
import copy
import xlwt

debug = False

class mlp_policyfunction(torch.nn.Module):
	def __init__(self, space_lenth, action_scale_list):
		super(mlp_policyfunction, self).__init__()
		self.space_lenth = space_lenth
		self.action_scale_list = action_scale_list
		self.fc1 = torch.nn.Linear(self.space_lenth + 1, 128)
		self.fc2 = torch.nn.Linear(128, 64)
		#layer fc3 is a multi-output mlp
		self.fc3 = list()
		for action_scale in self.action_scale_list:
			self.fc3.append(torch.nn.Linear(64, action_scale))
		self.fc3 = torch.nn.ModuleList(self.fc3)
	
	def forward(self, qfunction_input, dimension_index):
		input = torch.cat(
			(qfunction_input, 
			torch.tensor(dimension_index).float().view(1)),
			dim = -1
		)
		out1 = torch.nn.functional.relu(self.fc1(input))
		out2 = torch.nn.functional.relu(self.fc2(out1))
		out3 = self.fc3[dimension_index](out2)
		return torch.nn.functional.softmax(out3, dim = -1)

class RLDSE():
	def __init__(self, iindex):

		self.iindex = iindex

		seed = self.iindex * 10000
		#atype = int(self.iindex / 10)
		atype = 4

		torch.manual_seed(seed)
		numpy.random.seed(seed)
		random.seed(seed)

		#### step1 assign model
		self.config = my_test_config()
		#self.nnmodel = self.config.nnmodel
		self.has_memory = True	

		#### step2 assign platform
		#constrain
		self.constraints = self.config.constraints

		#### step3 assign goal
		#self.goal = self.config.goal
		self.config.config_check()

		self.workbook = xlwt.Workbook(encoding = 'ascii')
		self.worksheet = self.workbook.add_sheet("1")
		self.worksheet.write(0, 0, "period")
		self.worksheet.write(0, 1, "return")
		self.worksheet.write(0, 2, "loss")

		## initial DSE_action_space
		self.DSE_action_space = create_space_gem5()

		#define the hyperparameters
		self.PERIOD_BOUND = 500
		self.SAMPLE_PERIOD_BOUND = 1
		self.GEMA = 0.999 #RL parameter, discount ratio
		self.ALPHA = 0.001 #RL parameter, learning step rate
		self.THRESHOLD_RATIO = 2
		self.BATCH_SIZE = 1
		self.BASE_LINE = 0
		self.ENTROPY_RATIO = 0.1

		#initial mlp_policyfunction, every action dimension owns a policyfunction
		#TODO:share the weight of first two layer among policyfunction
		action_scale_list = list()
		for dimension in self.DSE_action_space.dimension_box:
			action_scale_list.append(int(dimension.get_scale()))
		self.policyfunction = mlp_policyfunction(self.DSE_action_space.get_lenth(), action_scale_list)

		##initial e_greedy_policy_function
		self.actor = actor_policyfunction()

		##initial evaluation
		#self.evaluation = evaluation_function(self.nnmodel, self.target)

		##initial optimizer
		self.policy_optimizer = torch.optim.Adam(
			self.policyfunction.parameters(), 
			lr=self.ALPHA, 
		)

		#### replay buffer, in order to record and reuse high return trace
		#### buffer is consist of trace list
		self.return_buffer = list()
		self.status_buffer = list()
		self.action_buffer = list()
		self.step_buffer = list()
		self.probs_noise_buffer = list()

		#### data vision related
		self.objectvalue_list = list()
		self.objectvalue_list.append(0)
		self.objectvalue_list2 = list()
		self.objectvalue_list2.append(0)

		self.objectvalue2_list = list()
		self.objectvalue2_list.append(0)

		self.period_list = list()
		self.period_list.append(-1)
		self.best_objectvalue = 10000
		self.best_objectvalue_list = list()
		self.best_objectvalue_list.append(self.best_objectvalue)

		self.all_objectvalue = list()
		self.all_objectvalue2 = list()
		self.all_objectvalue3 = list()

		self.best_objectvalue2 = 10000
		self.best_objectvalue2_list = list()
		self.best_objectvalue2_list.append(self.best_objectvalue)

		self.power_list = list()

		self.action_array = list()
		self.reward_array = list()

	def train(self):
		current_status = dict() #S
		next_status = dict() #S' 
		reward_log_name = "record/objectvalue/" + "reward_" + "ERDSE" + "_" + str(self.iindex) + ".txt"
		reward_log = open(reward_log_name, "w")

		period_bound = self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND
		for period in range(period_bound):
			print(f"period:{period}", end="\r")
			print(period)
			#here may need a initial function for action_space
			self.DSE_action_space.status_reset()

			#store status, log_prob, reward and return
			status_list = list()
			step_list = list()
			action_list = list()
			reward_list = list()
			return_list = list()
			probs_noise_list = list()

			batch_index = 0

			for step in range(self.DSE_action_space.get_lenth()): 
				step_list.append(step)
				#get status from S
				current_status = self.DSE_action_space.get_status()
				status_list.append(current_status)

				#use policy function to choose action and acquire log_prob
				action, probs_noise = self.actor.action_choose_with_no_grad(self.policyfunction, self.DSE_action_space, step)
				action_list.append(action)
				probs_noise_list.append(probs_noise)

				#take action and get next state S'
				next_status = self.DSE_action_space.sample_one_dimension(step, action)

				#### in MC method, we can only sample in last step
				#### and compute reward R
		
				#TODO:design a good reward function
				if(step < (self.DSE_action_space.get_lenth() - 1)): #delay reward, only in last step the reward will be asigned
					reward = float(0)
					reward2 = float(0)
				else:
					
					#pdb.set_trace()
					metrics = evaluation(next_status)

					if (metrics != None):
						energy =metrics['latency']
						area = metrics['Area']
						runtime = metrics['latency']
						power = metrics['power']
						self.constraints.update({"LATENCY": runtime,"POWER": power,"AREA": area})

						reward = 1000 / ( runtime*100000 * self.constraints.get_punishment())
						objectvalue = runtime
						objectvalue2 = power
						objectvalue3 = energy
						print(reward,self.constraints.get_punishment())
					else:
						reward = 0
					#### recording
					if(period < self.SAMPLE_PERIOD_BOUND):
						pass
					else:
						if(objectvalue < self.best_objectvalue  and self.constraints.is_all_meet()):
							self.best_objectvalue = objectvalue
							#print(f"best_status:{objectvalue, objectvalue2, power, DSP, BW, BRAM}")
						if (self.constraints.is_all_meet()):
							self.all_objectvalue.append(objectvalue)
							self.all_objectvalue2.append(objectvalue2)
							self.all_objectvalue3.append(objectvalue3)
						self.best_objectvalue_list.append(self.best_objectvalue)
						self.best_objectvalue2_list.append(self.best_objectvalue2)
						self.period_list.append(period - self.SAMPLE_PERIOD_BOUND)
						self.objectvalue_list.append(reward)
						self.objectvalue_list2.append(reward2)
						self.power_list.append(power)
						print(f"{period}\t{reward}", end='\n', file=reward_log)

				reward_list.append(reward)
				

				#assign next_status to current_status
				current_status = next_status

			self.action_array.append(self.DSE_action_space.get_action_list())
			self.reward_array.append(reward)

			#compute and record return
			return_g = 0
			T = len(reward_list)
			for t in range(T):
				return_g = reward_list[T-1-t] + self.GEMA * return_g
				return_list.append(torch.tensor(return_g).reshape(1))
			return_list.reverse()
			self.worksheet.write(period+1, 0, period)
			self.worksheet.write(period+1, 1, return_list[0].item())

			#### record trace into buffer
			self.use_max_value_expectation = True
			if(self.use_max_value_expectation):
				if(len(self.return_buffer) < 1):
					self.return_buffer.append(return_list)
					self.status_buffer.append(status_list)
					self.action_buffer.append(action_list)
					self.step_buffer.append(step_list)
					self.probs_noise_buffer.append(probs_noise_list)
				else:				
					min_index = numpy.argmin(self.return_buffer, axis = 0)
					min_index = min_index[0]
					min_return = self.return_buffer[min_index][0]
					if(return_list[0] > min_return):
						self.return_buffer[min_index] = return_list
						self.status_buffer[min_index] = status_list
						self.action_buffer[min_index] = action_list
						self.step_buffer[min_index] = step_list
						self.probs_noise_buffer[min_index] = probs_noise_list
					else:
						pass
			else:#### use high value trace replay
				if(return_list[0] > 8):
					self.return_buffer.append(return_list)
					self.status_buffer.append(status_list)
					self.action_buffer.append(action_list)
					self.step_buffer.append(step_list)
					self.probs_noise_buffer.append(probs_noise_list)
			

			if(period < self.SAMPLE_PERIOD_BOUND):
				pass
			elif(len(self.return_buffer) > 0):
				#### compute loss and update actor network
				loss = torch.tensor(0)
				for _ in range(self.BATCH_SIZE):
					#### random sample trace from replay buffer
					sample_index = random.randint(0, len(self.return_buffer)-1)
					s_return_list = self.return_buffer[sample_index]
					s_status_list = self.status_buffer[sample_index]
					s_action_list = self.action_buffer[sample_index]
					s_step_list = self.step_buffer[sample_index]
					s_probs_noise_list = self.probs_noise_buffer[sample_index]

					#### compute log_prob and entropy
					T = len(s_return_list)
					sample_loss = torch.tensor(0)
					pi_noise = torch.tensor(1)
					for t in range(T):
						s_entropy, s_log_prob = get_log_prob(self.policyfunction, self.DSE_action_space, s_status_list[t], s_action_list[t], s_step_list[t])
						retrun_item = -1 * s_log_prob * (s_return_list[t] - self.BASE_LINE)
						entropy_item = -1 * self.ENTROPY_RATIO * s_entropy
						sample_loss = sample_loss + retrun_item + entropy_item
						#pi_noise = pi_noise * s_probs_noise_list[t].detach()
					#### accumulate loss
					sample_loss = sample_loss / T
					loss = loss + sample_loss
				loss = loss / self.BATCH_SIZE
				#print(loss)


				#print(f"loss:{loss}")
				self.worksheet.write(int(period/self.BATCH_SIZE)+1, 2, loss.item())
				self.policy_optimizer.zero_grad()
				loss.backward()
				self.policy_optimizer.step()
			else:
				print("no avaiable sample")

		#end for-period
		self.workbook.save("record/new_reward&return/RLDSE_reward_record_old.xls")
	#end def-train

	def test(self):
		for period in range(1):
			# for step in range(self.DSE_action_space.get_lenth()):
			# 	action, _ = self.actor.action_choose_with_no_grad(self.policyfunction, self.DSE_action_space, step, is_train = False)
			# 	self.fstatus = self.DSE_action_space.sample_one_dimension(step, action)
			# self.evaluation.update_parameter(self.fstatus)
			# self.fruntime, t_L = self.evaluation.runtime()
			# self.fruntime = self.fruntime * 1000
			# self.fpower = self.evaluation.power()
			# print(
			# 	"\n@@@@  TEST  @@@@\n",
			# 	"final_status\n", self.fstatus,
			# 	"\nfinal_runtime\n", self.fruntime,
			# 	"\npower\n", self.fpower,
			# 	"\nbest_time\n", self.best_objectvalue_list[-1]
			# )
			pass


def run(iindex):
	print(f"%%%%TEST{iindex} START%%%%")

	DSE = RLDSE(iindex)
	print(f"DSE scale:{DSE.DSE_action_space.get_scale()}")
	DSE.train()
	DSE.test()

	workbook = xlwt.Workbook(encoding = 'ascii')
	worksheet = workbook.add_sheet("1")
	worksheet.write(0, 0, "index")
	worksheet.write(0, 1, "best_objectvalue")
	worksheet.write(0, 2, "best_objectvalue2")
	for index, best_objectvalue in enumerate(DSE.best_objectvalue_list):
		worksheet.write(index + 1, 0, index + 1)
		worksheet.write(index + 1, 1, best_objectvalue)
	for index, best_objectvalue in enumerate(DSE.best_objectvalue2_list):
		worksheet.write(index + 1, 2, best_objectvalue)
	name = "record/objectvalue/" +  "_" + "ERDSE" + "_" + str(iindex) + ".xls"
	workbook.save(name)
	workbook2 = xlwt.Workbook(encoding='ascii')
	worksheet2 = workbook2.add_sheet("1")
	worksheet2.write(0, 0, "index")
	worksheet2.write(0, 1, "objectvalue1")
	worksheet2.write(0, 2, "objectvalue2")
	for index, objectvalue in enumerate(DSE.all_objectvalue):
		worksheet2.write(index + 1, 0, index + 1)
		worksheet2.write(index + 1, 1, objectvalue)
	for index, objectvalue in enumerate(DSE.all_objectvalue2):
		worksheet2.write(index + 1, 2, objectvalue)
	for index, objectvalue in enumerate(DSE.all_objectvalue3):
		worksheet2.write(index + 1, 3, objectvalue)
	name = "record/objectvalue/"  + "ERDSE" + "_" + str(
		iindex) + "all_value" + ".xls"
	workbook2.save(name)
	'''
	tsne3D(DSE.action_array, DSE.reward_array, "ERDSE" + "_" + DSE.nnmodel + "_" + DSE.target)
	high_value_reward = 0
	for reward in DSE.reward_array:
		if(reward >= 10): high_value_reward += 1
	high_value_reward_proportion = high_value_reward/len(DSE.reward_array)
	hfile = open("high_value_reward_proportion_"+str(iindex)+".txt", "w")
	print(f"@@@@high-value design point proportion:{high_value_reward_proportion}@@@@", file=hfile)
	'''
	print(f"%%%%TEST{iindex} END%%%%")

if __name__ == '__main__':
	USE_MULTIPROCESS = False
	TEST_BOUND = 1

	if(USE_MULTIPROCESS):
		iindex_list = list()
		for i in range(TEST_BOUND):
			if(i < 10 or i >= 20): continue
			iindex_list.append(i)

		pool = Pool(1)
		pool.map(run, iindex_list)
		pool.close()
		pool.join()
	else:
		for iindex in range(TEST_BOUND):
			run(iindex)



