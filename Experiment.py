from itertools import product
from Problem import *
from Operators import *
from BinaryABC import BinaryABC
from AOS import *
from Utilities import Log, Logger
import sys
import time

class Experiment:
    def __init__(self, optimizer, operators, operator_selector, problem,  
    output_directory="results", type="test", algortihm_outs=None,  aos_outs = None,
    runs=30, clear_history = True ):
        self.optimizer = optimizer
        self.problem = problem
        self.operators = operators
        self.type = type
        self.clear_history = clear_history
        self.output_directory = output_directory
        self.aos_outs = aos_outs
        self.algortihm_outs = algortihm_outs
        self.runs = runs
        self.operator_selector = operator_selector
        self.optimizer.operator_selector = self.operator_selector
        self.optimizer.operator_pool = self.operators
        
        for o in self.operators:
            o.set_algorithm(self.optimizer)
        #Logging
        self.algorithm_logger = Logger(optimizer, algortihm_outs, output_directory, operator_selector.__conf__(), self.problem.ID)
        self.aos_logger = Logger(operator_selector, aos_outs, output_directory, operator_selector.__conf__(), self.problem.ID)
        
        if self.clear_history :
            self.algorithm_logger.clear_history()
            self.aos_logger.clear_history()

    def Run(self):
        self.times = []
        for run in range(self.runs):
            start_time = time.time()
            self.operator_selector.set_algorithm(self.optimizer,run)
            self.optimizer.run()
            self.times.append( time.time() - start_time) 
            self.algorithm_logger.log()
            self.aos_logger.log()
            self.optimizer.reset()
            

            



