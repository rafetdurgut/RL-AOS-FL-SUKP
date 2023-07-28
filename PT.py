from itertools import product
import os
import threading
import time

def thread_function(conf):
    time.sleep(1)
    print(conf)
    os.system(f"Python ./Run_Experiment.py {' '.join(map(str,conf.values()))}")
parameters = {"pNo":[4],"Method": ["average", "extreme"], "W": [5,25], "eps": [0.1,0.3,0.5], "alpha": [0.1, 0.5, 0.9],"gama": [0.1, 0.5, 0.9]}
configurations = [dict(zip(parameters, v)) for v in product(*parameters.values())]
for c in configurations:
    x = threading.Thread(target=thread_function, args=(c,))
    x.start()