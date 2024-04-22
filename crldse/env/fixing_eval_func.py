import math


class evaluation_function():
    def __init__(self, target):
        self.target = target

    def load_data_from_file(self, infile, k):
        pass

    def read_mapping(self, infile, k):
        pass

    def update_parameter(self, status, has_memory = False):
        pass

    def runtime(self):
        pass

    def power(self):
        pass

    def bandwidth(self):
        pass

    def Gops(self):
        pass

    def energy(self):
        pass
    
if __name__=='__main__':
    target = "normal"
    eval = evaluation_function(target=target)
    