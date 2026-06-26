import os
import numpy as np
from keller_segel_logistic import KellerSegel

def save_data(N, DU, DV, SENSITIVITY, R, CAPACITY, ALPHA, BETA, DT, seed, num_steps):
    sim = KellerSegel(N, DU, DV, SENSITIVITY, R, CAPACITY, ALPHA, BETA, DT, seed)
    history = sim.run(n_steps=num_steps, save_every=1000)
    steps, us, vs = zip(*history)
    folder_path = "data/runs/keller_segel_logistic/sensitivity_sweep"
    file_name = "chi="+str(sim.SENSITIVITY)+"_steps="+str(num_steps)+".npz"
    full_path = os.path.join(folder_path, file_name)
    np.savez(full_path, step=np.asarray(steps, dtype=object), u=np.asarray(us, dtype=object), v=np.asarray(vs, dtype=object))

for i in np.round(np.linspace(4.00, 4.30, 15), decimals=2):
    save_data(N=256, DU=0.2, DV=1, SENSITIVITY=i, R=0.1, CAPACITY=2, ALPHA=0.1, BETA=1, DT=0.1, seed=4, num_steps=50000)
#save_data(N=256, DU=0.2, DV=1, SENSITIVITY=4.1015, R=0.1, CAPACITY=2, ALPHA=0.1, BETA=1, DT=0.1, seed=4)