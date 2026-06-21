import os
import numpy as np
from keller_segel_logistic import KellerSegel

sim = KellerSegel(N=512, DU=1, DV=1, SENSITIVITY=5, R=0.05, CAPACITY=2, ALPHA=0.1, BETA=1, DT=0.1, seed=74)
history = sim.run(n_steps=1000, save_every=100)
steps, us, vs = zip(*history)

folder_path = "data/runs/keller_segel_logistic"
file_name = "chi="+str(sim.SENSITIVITY)+"_const"
full_path = os.path.join(folder_path, file_name)

np.savez(full_path, step=np.asarray(steps, dtype=object), u=np.asarray(us, dtype=object), v=np.asarray(vs, dtype=object))