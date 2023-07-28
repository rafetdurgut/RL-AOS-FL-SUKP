#%% Run Experiment
# Her bir region'da farkli credit degerleri olacak.
# Not only cluster center, But also cluster power.
import sys
from tkinter import W

from numpy import flip
from Experiment import Experiment
from Problem import *
from Operators import *
from BinaryABC import BinaryABC
from AOS import *
from CLRL import CLRL

# pNo = int(sys.argv[1])
# rewardType = sys.argv[2]
# W = int(sys.argv[3])
# eps = float(sys.argv[4])
# alpha = float(sys.argv[5])
# gama = float(sys.argv[6])

rewardType = "extreme"
W = 25
eps = 0.5
alpha = 0.5
gama = 0.3
pNo = 1
import pickle
model_rf = pickle.load(open('rf.sav', 'rb'))
model_svc = pickle.load(open('svc.sav', 'rb'))
model_mlp = pickle.load(open('mlp.sav', 'rb'))
models = [model_rf, model_svc, model_mlp]
for m in models:
    for p in range(0,30):
        problem= SetUnionKnapsack('Data/SUKP',pNo)
        # problem = OneMax(p)
        abc = BinaryABC(problem, pop_size=20, maxFE=max(problem.m, problem.n)*40, limit=100)

        # loadFileName="results/clusters-CLRL-4-extreme-0.5-25-0.5-0.3-1-Sphere.csv"
        loadFileName=None

        # problem = Rosenbrock(5)
        # problem = OneMax(1000)
        # problem = Weierstrass(30)
        # problem = Elliptic(30)
        # abc = BinaryABC(problem, pop_size=20, maxFE=250*40, limit=100)
        operator_pool = [ flipABC(), nABC(), ibinABC(), NBABC()]
        # operator_pool = [ , binABC(), ibinABC()]
        parameters = {'operator_size': len(operator_pool),"pNo":p,"reward_type": "extreme", "W": 25, "eps": 0.3, "alpha": 0.9,"gama": 0.5 ,"learning_mode":0,"load_file": None,"reward_func":0}
        # parameters = {'operator_size': len(operator_pool),"pNo":2500,"reward_type": "average", "W": 100, "eps": 0.3, "alpha": 0.5,"gama": 0.5,"learning_mode":0,"load_file": "CLRL-4-average-0.3-100-0.5-0.5-1-None-1-2500.csv","reward_func":1}
        operator_selectors = CLRL(parameters,m)
        alg_outs = ["convergence"]
        aos_outs = ["credits","rewards","usage_counter","success_counter","cluster_history","clusters","credit_history","reward_history","feature_information"]

        exp = Experiment(abc,operator_pool,operator_selectors,
        problem,algortihm_outs=alg_outs, aos_outs=[], runs=30, clear_history=True)

        exp.Run()