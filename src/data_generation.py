import os
import numpy as np
from keller_segel_logistic import KellerSegel

class DataGenerator:
    def __init__(self, N, DU, DV, SENSITIVITY, R, CAPACITY, ALPHA, BETA, DT, seed):
        self.N = N
        self.DU = DU
        self.DV = DV
        self.SENSITIVITY = SENSITIVITY
        self.R = R
        self.CAPACITY = CAPACITY
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.DT = DT
        self.seed = seed
    
    def save_densities(self, num_steps, save_every, sensitivity, file_path):
        sim = KellerSegel(self.N, self.DU, self.DV, sensitivity, self.R, self.CAPACITY, self.ALPHA, self.BETA, self.DT, self.seed)
        history = sim.run(n_steps=num_steps, save_every=save_every)
        steps, us, vs = zip(*history)
        np.savez(file_path, step=np.asarray(steps, dtype=object), u=np.asarray(us, dtype=object), v=np.asarray(vs, dtype=object))
    
    def multi_densities(self, num_steps, save_every, sensitivities, folder_name):
        file_paths = np.empty((sensitivities.shape))
        for i in sensitivities:
            folder_path = "data/runs/keller_segel_logistic/"+folder_name
            file_name = "chi="+str(i)+"_steps="+str(num_steps)+".npz"
            file_path = os.path.join(folder_path, file_name)
            file_paths.append(file_path)
            if os.path.isfile(file_path):
                continue
            self.save_densities(num_steps, save_every, i, file_path)
        return file_paths

    def save_div_steps(self, num_steps, div_threshold, sensitivities, file_path):
        div_steps = np.empty(sensitivities.shape)
        idx = 0
        for i in sensitivities:
            sim = KellerSegel(self.N, self.DU, self.DV, i, self.R, self.CAPACITY, self.ALPHA, self.BETA, self.DT, self.seed)
            div_steps[idx] = sim.run(n_steps=num_steps, div_threshold=div_threshold)
            print(str(i)+" done, "+str(div_steps[idx])+" steps")
            idx += 1
        np.savez(file_path, div_step=np.asarray(div_steps, dtype=object))

chi_values = np.round(np.linspace(4.1025, 4.1025, 1), decimals=5)
#chi_values = np.array([4.01, 4.03, 4.05, 4.07])

gen = DataGenerator(N=512, DU=0.2, DV=1, SENSITIVITY=None, R=0.1, CAPACITY=2, ALPHA=0.1, BETA=1, DT=0.1, seed=4)
#gen.save_div_steps(num_steps=50000, div_threshold=0.15, sensitivities=chi_values, file_path="data/runs/keller_segel_logistic/density_divergence/chi=4.1025aaaaa.npz")
gen.save_densities(10000, 100, 3, "data/runs/keller_segel_logistic/high_resolution_snapshots/chi=3.npz")