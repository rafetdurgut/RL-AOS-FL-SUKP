import numpy as np
import math
import copy


class AbstractOperator:
    # how many times did operator calls objective function
    def costFE(self):
        return 1

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def get_candidate(self, i, k):
        raise Exception("Should not call AbstractOperator!")


class binABC(AbstractOperator):
    def get_candidate(self, i, k):
        dim = np.random.randint(0, self.algorithm.problem.dimension - 1)
        temp = i.solution[dim] ^ k.solution[dim]
        if np.random.random() > 0.5:
            temp = not (i.solution[dim] ^ k.solution[dim])

        i.solution[dim] = temp ^ i.solution[dim]
        # it calls one time objective function. So our Abstractoperator returns it
        i.evaluate()
        return i
    def __str__(self):
        return "binABC"



class bitABC(AbstractOperator):
    def get_candidate(self, i, k):
        dim = np.random.randint(0, self.algorithm.problem.dimension - 1)
        phi = np.random.random() > 0.5
        temp = phi & (i.solution[dim] | k.solution[dim])
        i.solution[dim] = temp ^ i.solution[dim]
        # it calls one time objective function. So our Abstractoperator returns it
        i.evaluate()
        return i
    def __str__(self):
        return "bitABC"


class ibinABC(AbstractOperator):
    def __init__(self, qMax=0.3, qMin=0.1):
        super(ibinABC, self).__init__()
        self.qMax = qMax
        self.qMin = qMin

    def get_candidate(self, i, n):
        e_dim = int(np.random.random() * 0.1 * self.algorithm.problem.dimension +
                    math.exp(-4 * (self.algorithm.iteration / self.algorithm.max_iteration)) *
                    (0.1 * self.algorithm.problem.dimension) + 1)

        q = self.qMax - ((self.qMax - self.qMin) / self.algorithm.max_iteration) * (self.algorithm.iteration)
        if n.fitness > i.fitness:
            q = 0
        selected_dims = np.random.permutation(self.algorithm.problem.dimension)[0:e_dim]
        for d in selected_dims:
            if np.random.random() > q:
                i.solution[d] = i.solution[d] ^ (i.solution[d] ^ n.solution[d])
            else:
                i.solution[d] = i.solution[d] ^ (not (i.solution[d] ^ n.solution[d]))
        i.evaluate()
        return i
    def __str__(self):
        return "ibinABC"


class nABC(AbstractOperator):
    def get_candidate(self, i, k):
        minD = self.algorithm.problem.dimension // 20 + 1
        maxD = self.algorithm.problem.dimension // 10 + 1
        D = np.random.randint(minD, maxD)
        rand_perm = np.random.permutation(self.algorithm.problem.dimension)[0:D]
        i.solution[rand_perm] = k.solution[rand_perm]
        i.evaluate()
        return i
    def __str__(self):
        return "nABC"
    
class NBABC(AbstractOperator):
    def __init__(self, max_flips=0.1):
        super(NBABC, self).__init__()
        self.max_flips = max_flips
    def get_candidate(self, i, k):
        numD = int(math.ceil(self.max_flips*self.algorithm.problem.dimension))
        selected_dims = np.random.permutation(self.algorithm.problem.dimension)[0:numD]
        for d in selected_dims:
            if i.solution[d] != k.solution[d]:
                i.solution[d] =  k.solution[d]
        i.evaluate()
        return i

    def __str__(self):
        return "nBABC"

class BABC(AbstractOperator):
    def get_candidate(self, i, k):
        d = np.random.randint(0, self.algorithm.problem.dimension - 1)
        child1 = np.concatenate((i.solution[0:d], k.solution[d:]))
        child2 = np.concatenate((k.solution[0:d], i.solution[d:]))
        mutation_rate = 0.001 + 0.099 * np.random.random()
        rand_val = np.random.random(self.algorithm.problem.dimension) < mutation_rate
        child1 = np.bitwise_xor(child1, rand_val)
        child2 = np.bitwise_xor(child2, rand_val)
        # two times objective function call
        _, e1 = self.algorithm.problem.objective_function(child1)
        _, e2 = self.algorithm.problem.objective_function(child2)

        if e1 > e2:
            i.solution = copy.deepcopy(child1)
            i.cost = e1
        else:
            i.solution = copy.deepcopy(child2)
            i.cost = e2

        # just to update fitness value in bee.
        i.calculate_fitness()
        return i

    def costFE(self):
        return 2
    def __str__(self):
        return "BABC"


class disABC(AbstractOperator):
    def __init__(self, phiMax=0.9, phiMin=0.1):
        self.phiMax = phiMax
        self.phiMin = phiMin

    def get_candidate(self, i, n):
        phi = self.phiMax - ((self.phiMax - self.phiMin) / (self.algorithm.max_iteration)) * (self.algorithm.iteration)
        m1 = int(np.sum(i.solution))
        m0 = self.algorithm.problem.dimension - m1
        M11 = np.sum(np.logical_and(i.solution, n.solution))
        M10 = np.sum(np.logical_and(i.solution, np.logical_not(n.solution)))
        M01 = np.sum(np.logical_and(n.solution, np.logical_not(i.solution)))
        if (M11 + M01 + M10)==0:
            A = phi
        else:
            A = phi * (1 - (M11 / (M11 + M01 + M10)))
        z = 500000
        m11v = 0
        m10v = 0
        for k in range(m1):
            for j in range(m0):
                x = k / (m1 + j)
                zt = abs(1 - x - A)
                if zt < z:
                    z = zt
                    m11v = k
                    m10v = j
        Pxi = np.argwhere(i.solution == 1)
        Qxi = np.argwhere(i.solution == 0)
        Pxi = Pxi[:, 0]
        Qxi = Qxi[:, 0]
        bitsM11 = np.random.permutation(np.size(Pxi))[0:m11v]
        bitsM10 = np.random.permutation(np.size(Qxi))[0:m10v]
        i.solution[:] = False
        i.solution[Pxi[bitsM11]] = True
        i.solution[Qxi[bitsM10]] = True
        i.evaluate()
        return i
    def __str__(self):
        return "disABC"


class flipABC(AbstractOperator):
    def get_candidate(self, i, k):
        d1 = np.random.randint(0, self.algorithm.problem.dimension-1)
        i.solution[d1] = not(i.solution[d1])
        i.evaluate()
        return i
    def __str__(self):
        return "flipABC"


class GBABC(AbstractOperator):
    def _swapSolution(self, i):
        OnesPos = np.argwhere(i == 1)
        ZerosPos = np.argwhere(i == 0)
        if len(OnesPos) < 2 or len(ZerosPos) <2:
            return i
        d1 = np.random.randint(0, len(OnesPos) - 1)
        d2 = np.random.randint(0, len(ZerosPos) - 1)
        d1 = OnesPos[d1]
        d2 = ZerosPos[d2]
        temp = i[d1]
        i[d1] = i[d2]
        i[d2] = temp
        return i

    def get_candidate(self, i, k):
        zero = np.zeros(self.algorithm.problem.dimension, dtype=bool)
        neighbor2 = self.algorithm.neighbor_selection()
        while i == neighbor2 or k == neighbor2:
            neighbor2 = self.algorithm.neighbor_selection()
        pool = [i.solution, k.solution, self.algorithm.global_best.solution, zero, neighbor2.solution]
        selected = np.random.choice(5, 2)

        p1 = pool[selected[0]]
        p2 = pool[selected[1]]

        d1 = np.random.randint(3, self.algorithm.problem.dimension - 3)
        d2 = np.random.randint(3, self.algorithm.problem.dimension - 3)

        # Get Children using crossover
        child1 = np.concatenate(
            (p1[0:min(d1, d2)], p2[min(d1, d2):max(d1, d2)], p1[max(d1, d2):]))
        child2 = np.concatenate(
            (p2[0:min(d1, d2)], p1[min(d1, d2):max(d1, d2)], p2[max(d1, d2):]))

        # Swap Operator
        gchild1 = copy.deepcopy(child1)
        gchild2 = copy.deepcopy(child2)

        gchild1 = self._swapSolution(gchild1)
        gchild2 = self._swapSolution(gchild2)

        children = [child1, child2, gchild1, gchild2]
        objectives = []
        for c in children:
            c, y = self.algorithm.problem.objective_function(c)
            objectives.append(y)
        best_index = np.argmax(objectives)

        i.solution = copy.deepcopy(children[best_index])
        i.evaluate()
        return i

    def costFE(self):
        # there are 4 children for calculating objective_func
        return 4
    def __str__(self):
        return "GBABC"


class twoOptABC(AbstractOperator):
    def get_candidate(self, i, k):
        d1 = np.random.randint(0, self.algorithm.problem.dimension - 1)
        d2 = np.random.randint(0, self.algorithm.problem.dimension - 1)

        child1 = np.concatenate(
            (i.solution[0:min(d1, d2)], k.solution[min(d1, d2):max(d1, d2)], i.solution[max(d1, d2):]))
        child2 = np.concatenate(
            (k.solution[0:min(d1, d2)], i.solution[min(d1, d2):max(d1, d2)], k.solution[max(d1, d2):]))
        mutation_rate = 0.001 + 0.099 * np.random.random()
        child1, e1 = self.algorithm.problem.objective_function(child1)
        child2, e2 = self.algorithm.problem.objective_function(child2)
        if e1 > e2:
            i.solution = copy.deepcopy(child1)
        else:
            i.solution = copy.deepcopy(child2)
        i.evaluate()
        return i

    def costFE(self):
        # two children
        return 2
    def __str__(self):
        return "2optABC"
