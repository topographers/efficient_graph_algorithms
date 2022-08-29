
import os 
import time 
import sys 




class Evaluator():
    def __init__(self, device, f, method):
        #super(Evaluator, self).__init__()
        self.device = device
        self.f = f 
        self.method = method 
        
        self.forward_start_time = None 
        self.forward_end_time = None 
        self.forward_duration = None 
        #self.memory_snapshot = None 
        self.memory_usage = None 
        
    def start_time(self):
        self.forward_start_time = time.time() 
    def stop_time(self):
        self.forward_end_time = time.time() 
        self.forward_duration = self.forward_end_time - self.forward_start_time 
        
    def reset(self):
        self.forward_start_time = None 
        self.forward_end_time = None 
        self.forward_duration = None         
        #self.memory_snapshot = None 
        self.memory_usage = None         
        
    def record_memory_consumption(self, Mx):
        self.memory_usage = sys.getsizeof(Mx.storage()) * 1e-6
        #self.memory_usage = peak_memory
        
    #def record_memory_snapshot(self, snapshot):
    #    self.memory_snapshot = snapshot
        
    def print_statistics(self):
        print("device: {}".format(self.device))
        print("calculation time: {} ms \nstorage usage of Mx: {} MiB".format(self.forward_duration * 1e3, self.memory_usage))

        
        
        
        
        