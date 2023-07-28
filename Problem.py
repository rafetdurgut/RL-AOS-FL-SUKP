import struct
import numpy as np
from os import listdir
from os.path import isfile, join
from copy import deepcopy


class AbstractBaseProblem:
    bit_dimension= 20
    
    def objective_function(self):
        raise NotImplemented
    
    def binstrtosol(self,solution):
        temp_sol = []
        for i in range(int(self.dimension/self.bit_dimension)):
            temp = solution[i*self.bit_dimension:(i+1)*self.bit_dimension]
            sign = int(temp[0])
            p = temp[1:self.p+1]
            q = temp[self.p+1:]
            f =  sum(int(v)*(2**i) for i,v in enumerate(reversed(p)))
            qq = sum(int(v)*(1/(2**i)) for i,v in enumerate(q))
            val = (-1)**sign * (f+qq)
            temp_sol.append(val)
        return temp_sol


class OneMax(AbstractBaseProblem):
    def __init__(self, dimension):
        self.pname = 'OneMax'
        self.ptype = 1
        self.dosyaAdi = str(dimension)
        self.dimension = dimension
        self.ID = str(dimension)
        self.best = dimension
        #Temp vars
        self.m =100
        self.n=100

    def objective_function(self, solution):
        return solution, np.sum(solution)

class Task:
    def __init__(self, head, tail, s_cost, demand, dead_cost, inverse=-1):
        self.head = head
        self.tail=tail
        self.s_cost = s_cost
        self.demand=demand
        self.dead_cost=dead_cost
        self.inverse=inverse
        
class CARP(AbstractBaseProblem):
    MAX_NODE_NUM = 300
    def __init__(self, folderName, fileName):
        req_edge_num = nonreq_edge_num = 0
        self.tasks = []
        with open(f"DATA/CARP/{folderName}/{fileName}") as f:
            self.nov = int(f.readline().split()[0])
            self.noe = int(f.readline().split()[0])
            self.capacity = int(f.readline().split()[0])
            self.depot = int(f.readline().split()[0])
            
            self.trav_cost = np.full((self.nov+1, self.nov+1),np.inf)
            self.trav_cost[range(self.nov+1), range(self.nov+1)] = 0
            self.serv_cost = np.zeros((self.nov+1,self.nov+1))
            nonreq_edge_num = 0
            for i in range(self.nov):
                line = list(map(int, f.readline().split()))
                s_nod = line[0]
                for j in range(line[1]):
                    next_node = line[j*3 + 2]
                    self.trav_cost[s_nod][next_node] = line[j*3 + 3]
                    self.serv_cost[s_nod][next_node] = line[j*3 + 3]
                    if s_nod< next_node:
                        demand = line[j*3+4]
                        if demand > 0:
                            task_ = Task(s_nod, next_node, line[j*3 + 3], demand, line[j*3 + 3])
                            self.tasks.append(task_)
                        else:
                            nonreq_edge_num += 1
                            
            task_ = Task(self.depot, self.depot, 0, 0, 0, 0)
            req_edge_num = len(self.tasks)
            self.tasks.insert(0,task_)
            
            
            
            task_num = 2*req_edge_num
            
            
            for i in range(1, req_edge_num+1):
                self.tasks[i].inverse = i + req_edge_num
                
            for i in range(req_edge_num+1, 2*req_edge_num+1):
                print(i)
                task_ = Task(self.tasks[i-req_edge_num].tail, 
                             self.tasks[i-req_edge_num].head, 
                             self.tasks[i-req_edge_num].s_cost, 
                             self.tasks[i-req_edge_num].demand, 
                             self.tasks[i-req_edge_num].demand,
                             i-req_edge_num)
                self.tasks.append(task_)
                
            self.dijkstra()
            self.task_dist = np.zeros((task_num+1, task_num+1))
            for i in range(task_num+1):
                for j in range(task_num+1):
                    self.task_dist[i][j] = ((self.min_cost[self.tasks[i].head][self.tasks[j].head]) 
                    + (self.min_cost[self.tasks[i].head][self.tasks[j].tail])
                    + (self.min_cost[self.tasks[i].tail][self.tasks[j].head])
                    + (self.min_cost[self.tasks[i].tail][self.tasks[j].tail]))/4
                
            
                
            
            
            
    def dijkstra(self):
        sp = np.empty((self.noe+1, self.noe+1, self.noe+1), int)
        self.min_cost = np.full((self.noe, self.noe), np.inf)
        for i in range(1,self.noe+1):
            for j in range(1, self.noe+1):
                if i==j:
                    continue
                sp[i][j][0] = 1
                sp[i][j][1] = i
        mark = np.empty((self.MAX_NODE_NUM,), int)
        dist = np.empty((self.MAX_NODE_NUM,),float)
        dist1 = np.empty((self.MAX_NODE_NUM,),float)
        nearest_neighbor = np.empty((self.MAX_NODE_NUM,),int)
        
        for i in range(1, self.nov+1):
            mark[i] = 1
            for j in range(1,self.nov+1):
                if i==j:
                    continue
                
                mark[j] = 0
                dist[j] = self.trav_cost[i][j]
                dist1[j] = self.trav_cost[i][j]
            for k in range(1, self.nov+1):
                minimum = np.inf
                nearest_neighbor[0] = 0
                
                for j in range(1, self.nov+1):
                    if mark[j]:
                        continue
                    if dist1[j] == np.inf:
                        continue
                    if(dist1[j] < minimum):
                        minimum = dist1[j]
                if minimum == np.inf:
                    continue
                for j in range(1, self.nov+1):
                    if mark[j]:
                        continue
                    if dist1[j] == minimum:
                        nearest_neighbor[0]+=1
                        nearest_neighbor[nearest_neighbor[0]] = j
                v = nearest_neighbor[1]
                dist1[v] = np.inf
                mark[v] = 1
                
                if (sp[i][v][0] == 0) or (sp[i][v][0] > 0) and (sp[i][v][sp[i][v][0]] != v):    
                    sp[i][v][0] += 1
                    sp[i][v][sp[i][v][0]] = v
                
                for j in range(1,self.nov+1):
                    if mark[j]:
                        continue
                    
                    if (minimum + self.trav_cost[v][j]) <dist[j]:
                        dist[j] = minimum+self.trav_cost[v][j]
                        dist1[j] = minimum+self.trav_cost[v][j]
                    
                        for m in range(sp[i][v][0]+1):
                            sp[i][j][m] = sp[i][v][m]
                            
                for j in range(1, self.nov+1):
                    if i==j:
                        continue
                    self.min_cost[i][j] = dist[j]
            for i in range(1,self.nov+1):
                for j in range(1, self.nov+1):
                    if sp[i][j][0] == 1:
                        sp[i][j][0] == 0
                self.min_cost[i][i] = 0
                
            
                    
                
    def objective_function(self, solution):
        pass

class ZeroOneKnapsack(AbstractBaseProblem):
    def __init__(self, folderName, fileNo):
        mypath = folderName
        filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.dosyaAdi = filenames[fileNo]
        print(self.dosyaAdi)
        self.weights = []
        self.profits = []
        self.ID = self.dosyaAdi
        self.qualities = []
        self.non_qualities = []
        with open(f"{folderName}/{self.dosyaAdi}") as f:
            self.dimension, self.capacity = map(int, f.readline().split(' '))
            for i in range(self.dimension):
                line = f.readline().split(' ')
                self.weights.append(float(line[1]))
                self.profits.append(float(line[0]))
                self.qualities.append(float(line[0]) / float(line[1]))
                self.non_qualities.append(float(line[1]) / float(line[0]))

    'If the capacity is not fully filled'

    def optimizing_stage(self, solution):
        cap_val = np.sum(self.weights, where=solution)
        qualities = np.multiply(self.qualities, solution == 0)
        add_index = np.argmax(np.multiply(qualities, solution==0))
        while cap_val + self.weights[add_index] <= self.capacity:
            solution[add_index] = True
            qualities[add_index] = 0
            cap_val += self.weights[add_index]
            add_index = np.argmax(qualities)
        return solution

    'If the capacity is exceed'

    def repair(self, solution):
        cap_val = np.sum(self.weights, where=solution)
        qualities = np.multiply(self.non_qualities, solution)
        while cap_val > self.capacity:
            remove_index = np.argmax(qualities)
            solution[remove_index] = False
            qualities[remove_index] = 0
            cap_val -= self.weights[remove_index]
        return solution

    def objective_function(self, solution):
        cap_val = np.sum(self.weights, where=solution)
        if cap_val > self.capacity:
            solution = self.repair(solution)
            solution = self.optimizing_stage(solution)
        else:
            solution = self.optimizing_stage(solution)

        cap_val = np.sum(self.weights, where=solution)
        #print("cap:"+ str(cap_val))
        sum_val = np.sum(self.profits, where=solution)
        return solution, sum_val
        
class SetUnionKnapsack(AbstractBaseProblem):
    def __init__(self, folderName, fileNo):
        mypath = folderName
        self.ptype = 1
        filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.ID = filenames[fileNo]
        self.dosyaAdi = filenames[fileNo]
        f = open("{}/{}".format(folderName, self.dosyaAdi), "r")
        print("{}/{}".format(folderName, filenames[fileNo]))
        f.readline()
        f.readline()
        line1 = f.readline()
        start = line1.index('=')
        stop = line1.index(' ', start)
        self.m = int(line1[start + 1:stop])
        self.dimension = self.m
        start = line1.index('=', stop)
        stop = line1.index(' ', start)
        self.n = int(line1[start + 1:stop])
        start = line1.index('=', stop)
        iseof = line1.find(' ', start)
        if iseof == -1:
            stop = len(line1) - 1
        else:
            stop = line1.index(' ', start)

        self.C = int(line1[start + 1:stop])

        f.readline()
        f.readline()
        self.p = list(map(int, f.readline().split()))

        f.readline()
        f.readline()
        self.w = list(map(int, f.readline().split()))

        f.readline()
        f.readline()
        self.rmatrix = np.zeros((self.m, self.n), dtype=bool)
        self.items = []
        for i in range(self.m):
            self.items.append([])
            rm = list(map(int, f.readline().split()))
            self.rmatrix[i, :] = deepcopy(rm[:])
            self.items[i] = np.where(self.rmatrix[i][:] == True)
        self.R = np.zeros(self.m)
        self.freqs = np.sum(self.rmatrix, axis=0)
        for i in range(self.m):
            self.R[i] = self.p[i] / np.sum(self.w[j] / self.freqs[j] for j in range(self.n) if self.rmatrix[i][j])

        self.H = np.argsort(self.R)[::-1][:self.m]
        f.close()

    def optimizing_stage(self, solution, temp):
        trial = temp.copy()
        for i in range(self.m):
            a = self.H[i]
            if solution[a] == 0:
                b = self.items[a]
                trial[b] = 1
                cap_val = np.sum(self.w, where=trial)
                if cap_val <= self.C:
                    temp[b] = True
                    solution[a] = True
                else:
                    trial = temp.copy()

        return solution

    def repair(self, solution):
        temp_sol = np.zeros(self.m, dtype=bool)
        temp = np.zeros((self.n), dtype=bool)
        trial = np.array(temp, copy=True)

        for i in range(self.m):
            a = self.H[i]
            if solution[a]:
                b = self.items[a]
                trial[b] = True
                cap_val = np.sum(self.w, where=trial)
                if cap_val <= self.C:
                    temp[b] = True
                    temp_sol[a] = True
                else:
                    trial = temp.copy()
        return temp_sol, temp

    def objective_function(self, solution):
        temp = np.zeros((self.n), dtype=bool)
        for i in range(self.m):
            if solution[i]:
                temp[self.items[i]] = True
        cap_val = np.sum(self.w, where=temp)

        if cap_val > self.C:
            solution, temp = self.repair(solution)
            solution = self.optimizing_stage(solution, temp)
        else:
            solution = self.optimizing_stage(solution, temp)

        sum_val = np.sum(self.p[i] for i, val in enumerate(solution) if val)
        return solution, sum_val

class Sphere(AbstractBaseProblem):
    def __init__(self,dimension,bounds=[-100, 100]):
        self.name="sphere"
        self.ID=f"sphere-{dimension}"
        self.p=7
        self.dimension = dimension * 20
        self.bounds = bounds
    
    def objective_function(self,solution):
        obj = 0
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        obj = np.sum([val**2   for ind, val in enumerate(temp_sol)])
        return solution, obj
class Elliptic(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-100, 100]):
        self.name = "Elliptic"
        self.ID = f"Elliptic-{dimension}"
        self.p = 7
        self.dimension = dimension*20
        self.bounds = bounds
        
    def objective_function(self, solution):
        obj = 0
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        cost = np.sum( [val**2 * pow(10,6*((ind+1)/(len(temp_sol)-1)))  for ind, val in enumerate(temp_sol)] )
        return solution, cost

class SumSquares(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-10, 10]):
        self.name = "SumSquares"
        self.ID = f"SumSquares-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution,np.sum( np.linspace(1, len(temp_sol),len(temp_sol) ) * np.power(temp_sol,2))
    
class PowellSum(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-10, 10]):
        self.name = "PowellSum"
        self.ID = f"PowellSum-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution,np.sum( np.abs(np.power(temp_sol,np.linspace(1, len(temp_sol),len(temp_sol) ))))

class Scwefel222(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-10, 10]):
        self.name = "Scwefel222"
        self.ID = f"Scwefel222-{dimension}"

        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution,np.sum( np.abs(temp_sol)) +  np.prod(np.abs(temp_sol)) 
    
class Scwefel226(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-500, 500]):
        self.name = "Scwefel226"
        self.ID = f"Scwefel226-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution,418.98*len(temp_sol) - np.sum( temp_sol * np.sin(np.sqrt( np.abs(temp_sol) )) )
    
class Scwefel221(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-10, 10]):
        self.name = "Scwefel221"
        self.ID = f"Scwefel221-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        self.p = 4
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution, np.max(np.abs(temp_sol))

class Step(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-100, 100]):
        self.name = "Step"
        self.ID = f"Step-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution,np.sum( np.power( temp_sol + np.full(np.shape(temp_sol), 0.5), 2) )

class Quartic(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-1.28, 1.28]):
        self.name = "Quartic"
        self.ID = f"Quartic-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution,np.sum( np.linspace(1, len(temp_sol),len(temp_sol)) * np.power( temp_sol, 4) )
    
class QuarticWN(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-1.28, 1.28]):
        self.name = "QuarticWN"
        self.ID = f"QuarticWN-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution,np.random.random() + np.sum( np.linspace(1, len(temp_sol),len(temp_sol)) * np.power( temp_sol, 4) )
    
class Rosenbrock(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-10, 10]):
        self.name = "Rosenbrock"
        self.ID = f"Rosenbrock-{dimension}"
        self.dimension = dimension*20
        self.p = 4
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution, np.sum( [ 100*(temp_sol[i+1] - temp_sol[i]**2)**2 + (temp_sol[i]-1)**2  for i in range(len(temp_sol)-1)] )

class Rastrigin(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-5.12, 5.12]):
        self.name = "Rastrigin"
        self.ID = f"Rastrigin-{dimension}"
        self.p = 3
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution,np.sum( [ i**2 - 10*np.cos(2*np.pi*i) +10   for i in temp_sol] )
    
class NCRastrigin(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-5.12, 5.12]):
        self.name = "NCRastrigin"
        self.ID = f"NCRastrigin-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def y(self, i):
        if np.abs(i) < 0.5:
            return i
        return round(2*i)/2
    
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution,np.sum( [ self.y(i)**2 - 10*np.cos(2*np.pi*self.y(i)) +10   for i in temp_sol] )

class Griewank(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-600, 60]):
        self.name = "Griewank"
        self.ID = f"Griewank-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        return solution,1 + np.sum( [1/400* (x**2) for i,x in enumerate(temp_sol)] ) + np.prod( [np.cos(x / np.sqrt(i+1)) for i,x in enumerate(temp_sol)] )

class Ackley(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-32, 32]):
        self.name = "Ackley"
        self.ID = f"Ackley-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        n= len(temp_sol)
        a = (1/n)*np.sum(np.power(temp_sol,2))
        b = (1/n)*np.sum([ np.cos(2*np.pi*i) for i in temp_sol])
        return solution,-20*np.exp(-0.2 * np.sqrt(a ) ) -  np.exp(b) + 20 + np.e
    
class Penalized1(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-50, 50]):
        self.name = "Penalized1"
        self.ID = f"Penalized1-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def u(self,xi,a,k,m):
        if xi>a:
            return k*(xi-a)**m
        if xi>=-a and xi<=a:
            return 0
        return k*(-xi-a)**m
    
    def y(self,xi):
        return 1+0.25*(xi+1)
    
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        a=10
        k=100
        m=4
        n = len(temp_sol)
        total_u = np.sum([self.u(temp_sol[i], a, k, m) for i in range(n)])
        yi = [self.y(i) for i in temp_sol]
        term = np.sum([ (yi[i]-1)**2 * (1 + 10*pow(np.sin(np.pi*yi[i+1]),2) ) for i in range(n-1) ] )
        result=(np.pi/n)*(10*pow(np.sin(np.pi*yi[0]),2) + term + pow((yi[n-1]-1),2)) + total_u;
        return solution,result
    
class Penalized2(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-50, 50]):
        self.name = "Penalized2"
        self.ID = f"Penalized2-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def u(self,xi,a,k,m):
        if xi>a:
            return k*(xi-a)**m
        if xi>=-a and xi<=a:
            return 0
        return k*(-xi-a)**m
    
    def y(self,xi):
        return 1+0.25*(xi+1)
    
    def objective_function(self, solution):
        a=10
        k=100
        m=4
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        n = len(temp_sol)
        total_u = np.sum([self.u(temp_sol[i], a, k, m) for i in range(n)])
        term = np.sum([ (temp_sol[i]-1)**2 * (1 + pow(np.sin(3*np.pi*temp_sol[i+1]),2) ) for i in range(n-1) ] )
        result=(np.pi/n)*(10*pow(np.sin(np.pi*temp_sol[0]),2) + term + pow((temp_sol[n-1]-1),2))*(1+pow(np.sin(np.pi*temp_sol[n-1]),2)) + total_u;
        return solution,result
    
class Alpine(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-10, 10]):
        self.name = "Alpine"
        self.ID = f"Alpine-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])

        return solution,np.sum( np.abs(temp_sol*np.sin(temp_sol) ) + 0.1*np.asarray(temp_sol)) 

class Levy(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-10, 10]):
        self.name = "Levy"
        self.ID = f"Levy-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        result=0;
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])

        n = len(temp_sol)
        for p in range(n-1):
            result += pow(temp_sol[p]-1,2)*(1+pow(np.sin(3*np.pi*temp_sol[p+1]),2))
        result += (pow(np.sin(3*np.pi*temp_sol[0]),2)) + (np.abs(temp_sol[n-1])* (1+(pow(np.sin(3*np.pi*temp_sol[n-1]),2))))
        return solution,result

class Schaffer(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-10, 10]):
        self.name = "Schaffer"
        self.ID = f"Schaffer-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        result1=0
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])

        n = len(temp_sol)
        for p in range(n-1):
             result1 += pow(temp_sol[p],2);
        result = 0.5 + ( (pow(np.sin(np.sqrt(result1)),2)-0.5) / (pow((1+0.001*result1),2)) );
        return solution,result
    
class Weierstrass(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-0.5, 0.5]):
        self.name = "Weierstrass"
        self.ID = f"Weierstrass-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        self.p = 0
        
    def objective_function(self, solution):

        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        val=0
        temp=0
        result=0
        n = len(temp_sol)
        for p in range(n):
            for k in range(20):
                 val+=pow(0.5,k)*np.cos(2*np.pi*pow(3,k)*(temp_sol[p]+0.5));
        
        for k in range(20):
            temp+=pow(0.5,k)*np.cos(2*np.pi*pow(3,k)*0.5);
        
        result=val-n*temp;
        return solution, result;
    
class Himmelblau(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[-5, 5]):
        self.name = "Himmelblau"
        self.ID = f"Himmelblau-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        result=0
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        n = len(temp_sol)
        for p in range(n):
            result += (pow(temp_sol[p],4) - 16*pow(temp_sol[p],2) + 5*temp_sol[p]);
        return solution,result/n
    
class Michalewicz(AbstractBaseProblem):
    def __init__(self, dimension, bounds=[0,np.pi]):
        self.name = "Michalewicz"
        self.ID = f"Michalewicz-{dimension}"
        self.dimension = dimension*self.bit_dimension
        self.bounds = bounds
        
    def objective_function(self, solution):
        result=0
        temp_sol = self.binstrtosol(solution)
        temp_sol = np.clip(temp_sol,self.bounds[0], self.bounds[1])
        m=10
        n = len(temp_sol)
        for p in range(n):
            result+= np.sin(temp_sol[p])*pow(np.sin(((p+1)*pow(temp_sol[p],2))/np.pi),(2*m));
        return solution,-result