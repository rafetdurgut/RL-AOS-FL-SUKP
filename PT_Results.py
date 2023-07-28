import csv
from itertools import product
from Problem import SetUnionKnapsack
def get_best_data(fileName, operator_size):
    datas = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        previous_iter = 0
        previous_val = 0
        for row in csv_reader:
            if len(row) > 0:
                iteration, val = row
                if iteration < previous_iter:
                    datas.append((previous_val))
                else:
                    previous_val = val
                previous_iter = iteration
        datas.append((previous_val))
    return datas
import numpy as np
# parameters = {"Method": ["average", "extreme"], "W": [5, 25], "Pmin": [0.1, 0.2], "Alpha": [0.1, 0.5, 0.9]}
parameters = {"pNo":[15],"Method": ["average", "extreme"], "W": [5,25], "eps": [0.1,0.3,0.5], "alpha": [0.1, 0.5, 0.9],"gama": [0.1, 0.5, 0.9]}

configurations = [dict(zip(parameters, v)) for v in product(*parameters.values())]
ind = 0
data_maks = []
data_means = []

filenames=[]
for c in configurations:
    problem=SetUnionKnapsack('Data/SUKP',4)
    filenames.append(problem.dosyaAdi)
    file_name = f"results/convergence-CLRL-3-{c['Method']}-{c['eps']}-{c['W']}-{c['alpha']}-{c['gama']}-0-sukp 300_300_0.15_0.85.txt.csv"
    data =get_best_data(file_name, 3)
    data_maks.append(max(data))
    data_means.append(np.mean(data))
    ind += 1
print(data_maks)
print(data_means)

