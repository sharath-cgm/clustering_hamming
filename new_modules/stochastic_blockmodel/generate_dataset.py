import numpy as np

def generate_block(m, n, f):
	# returns a block size m*n with f fraction of 1s
	return np.random.choice([0,1], size = (m,n), p = [1-f, f])

def generation(N, D, k, p = 0.4, q = 0.1):

	data = generate_block(N, D, q)
	labels = []
	temp = 0

	block_rows, block_columns = int(N/k), int(D/k)
	for i in range(k):
		data[i*block_rows:(i+1)*block_rows, i*block_columns:(i+1)*block_columns] = generate_block(block_rows, block_columns, p)

		label_tiled = np.tile([i], (1, block_rows))
		if temp == 0:
			temp = 1
			labels = label_tiled
		else:
			labels = np.concatenate((labels, label_tiled), axis = 1)

	data = np.concatenate((data, labels.transpose()) , axis = 1)

	return data

# N, D, k = 35, 14, 7
# data = generation(N,D,k, p = 1, q = 0)

# N, D, k = 10000, 100, 5
N, D, k = 100000, 500, 50
# data = generation(N,D,k, p = 1, q = 0)
data = generation(N, D, k)

filename = "N" + str(N) + "_D" + str(D) + "_k" + str(k) + "_1"
np.savetxt(filename + ".txt", data, fmt = '%u', delimiter=" ")

unique_entries, _ = (np.unique(data, axis=0)).shape
print(unique_entries)

# print(data)