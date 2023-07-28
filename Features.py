import numpy as np
class Features:
    def __init__(self,ptype=1):
        self.ptype = ptype
        pass
    def population_features(self,parents,childs,last_best, new_best ,n,dimension,max_trial, t):
        features = []
        c=[(i,j) for i in range(n-1) for j in range(i+1,n)]

        imp_childs = []
        best_childs = []
        if self.ptype == 0:
            for ind,p in enumerate(childs):
                if p.cost < parents[ind].cost:
                    #Population improving childs
                    imp_childs.append(p)
                if p.cost < last_best.cost:
                    best_childs.append(p)
        elif self.ptype == 1:
            for ind,p in enumerate(childs):
                if p.cost > parents[ind].cost:
                    imp_childs.append(p)
                if p.cost > last_best.cost:
                    best_childs.append(p)

        #Population solution diversity
        dists = [self.hamming_distance(parents[i].solution,parents[j].solution) for i,j in c ]
        psd = np.mean(dists)/dimension
        pdd = np.max(dists) / dimension
        #Population Fitness diversity
        pfd = np.mean([abs(parents[i].fitness-parents[j].fitness) for i,j in c ])
        #Population improving childs
        pic = len(imp_childs)/n
        #Population new best childs
        pnb = len(best_childs)/n
        #Population amount of improvement
        
        #Population Convergence Velocity
        #Ptype == 0 means minimization
        if self.ptype == 0:
            pai = 0
            if len(imp_childs)>0:
                pai = np.mean([ (p.cost - parents[ind].cost )/parents[ind].cost for ind,p in enumerate(imp_childs) if parents[ind].cost!= 0])/n
            a=min(parents, key=lambda b: b.cost)
            b=min(childs, key=lambda b: b.cost)
            pcv = (a.cost - b.cost)/a.cost
            pcr =  abs(self.hamming_distance(new_best.solution, a.solution) - self.hamming_distance(new_best.solution, b.solution))/dimension

        else:
            pai = 0
            if len(imp_childs)>0:
                pai = np.mean([ (parents[ind].cost - p.cost)/parents[ind].cost for ind,p in enumerate(imp_childs) if parents[ind].cost!= 0])/n
            a=max(parents, key=lambda b: b.cost)
            b=max(childs, key=lambda b: b.cost)
            pcv = (b.cost - a.cost)/a.cost
            pcr =  abs(self.hamming_distance(new_best.solution, a.solution) - self.hamming_distance(new_best.solution, b.solution))/dimension
        #Evolvability probability
        eap = 0
        if len(imp_childs)>0:
            std = np.std(list(map(lambda x: x.cost, parents)))
            if std != 0:
                eap = np.mean([(abs(last_best.cost - s.cost)/n)/std for s in imp_childs])
        
        #Average Trial Number
        atn = np.mean([a.trial for a in parents])/max_trial

        features.append(psd)
        features.append(pfd)
        features.append(pic)
        # features.append(pwc)
        # features.append(pec)
        features.append(pnb)
        features.append(abs(pai))
        features.append(pcv)
        features.append(pcr)
        # print(pcr)
        features.append(eap)
        features.append(eap*pic)
        features.append(atn)
        features.append(pdd)
        return features
        
    def hamming_distance(self,sol1,sol2):
        return np.count_nonzero(sol1!=sol2)
