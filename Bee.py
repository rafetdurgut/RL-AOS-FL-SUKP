import numpy as np


class Bee:
    def __init__(self, problem=None, solution=None):
        self.problem = problem
        self.fitness = 0
        self.cost = 0
        self.prev_solution = []
        self.prev_cost = None
        self.trial = 0
        self.normalize = 1
        self.features = []
        
        if solution is None:
            self.solution = np.random.random(self.problem.dimension) > 0.5
        else:
            self.solution = solution
        self.evaluate()

    def calculate_features(self, colony, osr):
        #Solution Distance
        
        if not (colony.features):
            return
        self.features = np.zeros((7,))
        self.features[0] = np.count_nonzero(colony.global_best.solution != self.prev_solution)/self.problem.dimension       #How many bits had been far away from global best
        self.features[1] = np.count_nonzero(self.solution != self.prev_solution)/self.problem.dimension                     #How many bits had been change
        #Objective Distance
        if self.problem.ptype == 0:
            self.features[2] = colony.global_best.cost - self.prev_cost/colony.global_best.cost                        #How close to global best
            self.features[3] = (self.prev_cost - self.cost)/self.prev_cost                                             #How much improved                              
            cbest=min(colony.colony, key=lambda b: b.cost)  
            cworst=max(colony.colony, key=lambda b: b.cost)
            
        else:
            self.features[2] = abs(colony.global_best.cost - self.prev_cost)/colony.global_best.cost
            self.features[3] = abs(self.prev_cost - self.cost)/self.prev_cost
            cbest=max(colony.colony, key=lambda b: b.cost)
            cworst=min(colony.colony, key=lambda b: b.cost)

        self.features[4] = np.count_nonzero(cbest.solution != self.prev_solution)/self.problem.dimension
        self.features[5] = np.count_nonzero(cworst.solution != self.prev_solution)/self.problem.dimension
        #Trial Distance
        self.features[6] = self.trial/colony.limit
        self.features = np.append(self.features,colony.features)
        self.features = np.append(self.features,[osr])
        self.feature_size = len(self.features)

    def evaluate(self):
        self.solution, self.cost = self.problem.objective_function(self.solution)
        self.calculate_fitness()
        

    def initial(self):
        self.trial = 0
        self.solution = np.random.random(self.problem.dimension) > 0.5
        self.evaluate()

    def get_better(self, candidate):
        if candidate.cost < self.cost and self.problem.ptype==0:
            candidate.trial = 0
            return candidate
        elif candidate.cost > self.cost and self.problem.ptype==1:
            candidate.trial = 0
            return candidate
        else:
            self.trial += 1
            return self

    def __str__(self):
        return f'Trial:{self.trial}, Cost:{self.cost}'

    def calculate_fitness(self):
        if (self.cost>0):
            self.fitness =(1/(1+self.cost))
        else:
            self.fitness = 1 + abs(self.cost)
