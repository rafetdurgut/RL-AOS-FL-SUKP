from cmath import inf
import numpy as np
import pandas as pd
#Learning mode : 0 ; Starts from zero for each runtime.
#Learning mode : 1 ; Learn from first runtime and then freeze.
#Learning mode : 2 ; Continuously learn for all runtime.

class abstractOperatorSelection:
    def __init__(self, operator_size, reward_type, W=5, alpha=0.1, beta=0.5, Pmin=0.1, learning_mode=0, load_from_file=None):
        self.learning_mode = learning_mode
        self.operator_size = operator_size
        self.rewards = [[0] for _ in range(self.operator_size)]
        self.rewards_history =  np.zeros((operator_size))
        self.credits = [[0] for _ in range(self.operator_size)]
        self.credits_history = [[] for _ in range(self.operator_size)]
        self.success_counter = [[0] for _ in range(operator_size)]
        self.total_succ_counters = np.zeros((operator_size))
        self.usage_counter = [[0] for _ in range(operator_size)]
        self.probabilities = np.zeros((operator_size))
        self.reward = np.zeros((operator_size))
        self.type = 'iteration'
        self.iteration = 0
        self.reward_type = reward_type
        self.W = W
        self.Pmin = Pmin
        self.eps = Pmin
        self.Pmax = 1 - (self.operator_size - 1) * Pmin
        self.alpha = alpha
        self.beta = beta
        self.operator_informations = []
        self.runtime = 0
        self.auto_mode = False
        self.max_period = 5
        self.current_period = 0
        self.mix_mode = False
        self.feature_information = []

    def reset(self):
        self.rewards = [[0] for _ in range(self.operator_size)]
        self.number_of_updates = [0 for i in range(self.operator_size)]
        self.rewards_history =  np.zeros((self.operator_size))
        self.credits = [[0] for _ in range(self.operator_size)]
        self.credits_history = [[] for _ in range(self.operator_size)]
        self.success_counter = [[0] for _ in range(self.operator_size)]
        self.total_succ_counters = np.zeros((self.operator_size))
        self.usage_counter = [[0] for _ in range(self.operator_size)]
        self.probabilities = np.zeros((self.operator_size))
        self.reward = np.zeros((self.operator_size))
        self.Pmax = 1 - (self.operator_size - 1) * self.Pmin
        #print(self.algorithm.feature_size)
        if self.load_from_file == None:
            self.clusters = np.full((self.max_period,self.operator_size, self.algorithm.feature_size),np.inf)
        else:
            with open(self.load_from_file, 'r',) as f:
                row = f.readlines()
                
                clusters = []
                for r in row:
                    r = r.replace('"',"");
                    cl = r.split(',')
                    for k in range(len(cl)):
                        cc= np.fromstring(cl[k][1:-1], dtype=np.float32, sep=' ')
                        clusters.append(cc)
            a = clusters[-self.max_period*self.operator_size:]
            self.clusters = np.reshape(a,(self.max_period,self.operator_size,self.algorithm.feature_size))

            

    def set_algorithm(self, algorithm,run_number):
        self.iteration = 0
        #print(run_number)
        self.operator_informations = []
        self.timer = np.zeros((self.operator_size),dtype=int)
        self.cluster_history = [[] for _ in range(self.operator_size)]
        self.run_number = run_number;
        self.algorithm = algorithm
        if (self.learning_mode == 1 and run_number > 0):
            self.auto_mode = True
        else:
            self.reset()
        
        
        

    def get_reward(self, candidate_cost, old_fitness):
        if(candidate_cost==0):
            return 0
        r = float((  candidate_cost - old_fitness )) * (self.algorithm.global_best.cost/candidate_cost)
        # r = float(( old_fitness > new_fitness))
        # Add reward to rewards
        # Update Credits ...
        if r < 0:
            r = 0
        return r

    def next_period(self):
        pass
    def next_iteration(self):
        self.iteration += 1
        o_period = self.current_period
        self.current_period = int(np.floor(self.iteration/(self.algorithm.max_iteration/self.max_period)))
        self.eps = self.Pmin * np.exp(-4*self.iteration/self.algorithm.max_iteration)
        if(o_period != self.current_period):
            self.next_period()
        
        for i in range(self.operator_size):
            self.credits_history[i].append(self.credits[i][-1])
            self.operator_informations.append([i, self.run_number, self.algorithm.global_best.cost, self.iteration, self.credits_history[i][-1], 
            self.rewards_history[i], self.usage_counter[i][-1], self.success_counter[i][-1]])
            # self.rewards[i].append(0)
            self.usage_counter[i].append(0)
            self.rewards_history[i] = 0
            self.success_counter[i].append(0)

    def add_reward(self, op_no, candidate, current):
        self.usage_counter[op_no][-1] += 1
        reward = self.get_reward(candidate.cost, current.cost)
        if reward > 0:

            self.success_counter[op_no][-1] += 1
            self.total_succ_counters[op_no] += 1
            candidate.calculate_features(self.algorithm)
            if len(self.algorithm.features)>0:
                op_f = [self.success_counter[op_no][-1]/self.usage_counter[op_no][-1], op_no]
                self.feature_information.append(np.concatenate((self.algorithm.features, candidate.features,op_f)))


    def apply_rewards(self, i):
        if self.reward_type == "insta":
            reward = self.rewards[i][self.iteration]
        elif self.reward_type == "average":
            start_pos = max(0, len(self.rewards[i]) - self.W)
            reward = np.average(self.rewards[i][start_pos:len(self.rewards[i])])
        elif self.reward_type == "extreme":
            start_pos = max(0, len(self.rewards[i]) - self.W)
            reward = np.max(self.rewards[i][start_pos:len(self.rewards[i])])
        return reward

    def update_credits(self,op_no,distance=0):
        r = self.apply_rewards(op_no)
        if(distance==inf):
            distance =0
        credit = (1 - self.alpha) * self.credits[op_no][-1] + self.alpha * (r+self.gama*distance)
        self.credits[op_no].append(credit)

    def operator_selection(self, candidate=None):
        raise Exception("Should not call Abstract Class!")

    def roulette_wheel(self, ):
        sumProbs = sum(self.probabilities)
        probs = [item / sumProbs for item in self.probabilities]
        op = np.random.choice(len(probs), p=probs)
        return op


class ClusterRL(abstractOperatorSelection):
    def __init__(self, operator_size, reward_type, W, alpha,  Pmin, gama = 0.3,learning_mode=0,load_from_file=None):
        super(ClusterRL, self).__init__(operator_size, reward_type, W, alpha=alpha, beta=0.1, Pmin=Pmin,learning_mode=learning_mode)
        self.operator_size = operator_size
        self.learning_mode = learning_mode
        self.alpha = alpha
        self.type = 'function'
        self.gama = gama
        self.timer = np.zeros((self.operator_size),dtype=int)
        self.cluster_history = [[] for _ in range(self.operator_size)]
        self.n_cluster_history = np.zeros( (self.max_period, self.operator_size) )
        self.load_from_file=load_from_file

    def get_reward_bytype(self,op_no):
        if self.timer[op_no] == 0:
            return self.rewards[op_no][-1]
        if self.reward_type == "insta":
            return self.rewards[op_no][-1]
        elif self.reward_type == "average":
            start_pos = max(0, len(self.rewards[op_no]) - self.W)
            reward = np.sum(self.rewards[op_no][start_pos:-1])/(self.iteration-start_pos)
            return  reward
        elif self.reward_type == "extreme":
            start_pos = max(0, len(self.rewards[op_no]) - self.W)
            reward = np.max(self.rewards[op_no][start_pos:-1])
            return reward

    def add_reward(self, op_no, candidate, current):
        super(ClusterRL, self).add_reward(op_no, candidate, current)
        reward = self.get_reward(candidate.cost, current.cost)
        self.rewards_history[op_no] += reward
        self.rewards[op_no].append(reward)
        self.update_credits(op_no,self.distance(op_no, current))
        # r = reward + self.gama * self.distance(op_no, current)
        if reward > 0:
            self.update_cluster(op_no, current)
            self.timer[op_no] += 1
        
        #self.rewards[op_no].append(reward)
            
            #self.iter_rewards[op_no][self.iteration] += reward
            #self.iter_credits[op_no][self.iteration] += credit
            #self.credits[op_no].append(credit)
            #self.iter_rewards[op_no][self.iteration] += r + self.gama * self.hamming_distance(self.clusters[op_no], candidate.solution)

    def update_cluster(self, op, current):
        
        if len(current.features) == 0:
            return
        if np.sum(self.clusters[self.current_period][op][:]) == np.inf :
            for i in range( len(current.features) ) :
                self.clusters[self.current_period][op][i] = current.features[i]
            return
        for i in range(len(current.features) ) :
            self.clusters[self.current_period][op][i] = self.clusters[self.current_period][op][i] + (current.features[i]-self.clusters[self.current_period][op][i]) / (
                        self.n_cluster_history[self.current_period][op] + 1)
            self.n_cluster_history[self.current_period][op] += 1
        
        self.cluster_history.append(np.concatenate(([op, self.iteration, self.run_number], self.clusters[self.current_period][op])))

    
        
    def distance(self, op, candidate):
        dist = 0

        if self.iteration==0:
            return dist
        

        return  (1/len(self.algorithm.features))*np.linalg.norm(self.clusters[self.current_period][op] - candidate.features)
        
        
    def operator_selection(self, candidate):
        #Epsilon-greedy
        if np.random.rand() < self.eps or self.learning_mode == -1:
            for i in range(self.operator_size):
                self.probabilities[i] = 1
            return self.roulette_wheel()

        #AOS
        credits = [self.credits[ind][-1] for ind in range(self.operator_size)]
        if(np.std(credits)!= 0):
            credits = (credits - np.min(credits))/(np.max(credits)-np.min(credits))

        if self.auto_mode:
            a = np.exp(-4*self.iteration/self.algorithm.max_iteration)
            a = -1
            values = [-a * self.credits[ind][-1] + self.gama* self.distance(ind, candidate) for ind in range(self.operator_size)]
        elif self.mix_mode:
            a = np.exp(-5*self.iteration/self.algorithm.max_iteration)
            a = -1
            values = [-a * self.credits[ind][-1] + self.gama * self.distance(ind, candidate) for ind in range(self.operator_size)]
        else:
            a = np.exp(-5*self.iteration/self.algorithm.max_iteration)
            a = -1
            values = [-a * self.credits[ind][-1] + self.gama * self.distance(ind, candidate) for ind in range(self.operator_size)]
        
        best_op = np.argmin(values)
        return best_op

    def __conf__(self):
        return ['CLRL', self.operator_size,  self.reward_type, self.Pmin, self.W, self.alpha, self.gama,self.learning_mode]

