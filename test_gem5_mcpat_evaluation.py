from gem5_mcpat_evaluation import evaluation


cheackdik = dict()
cheackdik["core"] = 4
cheackdik["l1i_size"] = 8
cheackdik["l1d_size"] = 8
cheackdik["l2_size"] = 8
cheackdik["l1d_assoc"] = 2
cheackdik["l1i_assoc"] = 2
cheackdik["l2_assoc"] = 2
cheackdik["sys_clock"] = 2.8
metrics = evaluation(cheackdik)
