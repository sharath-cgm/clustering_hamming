import numpy as np

data = np.loadtxt("N_30.txt", dtype=float)
data = data.astype(int)
labels = data[:, -1]
data = data[:, 0:-1]

