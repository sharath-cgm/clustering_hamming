import numpy as np

def check_repetitions(X, out = False):
	N, _ = X.shape
	unique_entries, _ = (np.unique(X, axis=0)).shape

	if out == True:
		print(N, unique_entries)
		
	if N == unique_entries:
		return True

	return False


def convert(X, hamming_dim):
	N, D = X.shape
	X = X.astype(np.float32)
	X_mean = X.mean(axis=0)
	X -= X_mean

	hamming_data = np.zeros((N, hamming_dim), dtype = int)

	for i in range(hamming_dim):
		# generate D-dimensional 'hamming_dim' no. of random vectors (each element chosen uniformly at random in range [-1, 1) ).
		rand_normal = np.random.uniform(-1, 1, D)

		for j in range(N):
			# take inner product, if positive- 1, else 0
			product = np.dot(X[j], rand_normal)
			if product > 0:
				hamming_data[j][i] = 1

	return hamming_data

def euclidean_to_hamming(X, k):
	for _ in range(15):
		hamming_data = convert(X, k+15) # hamming_dim = k+1
		if check_repetitions(hamming_data):
			break

	return hamming_data



# N = 10
# D = 2
# k = 5

# data = np.array(
# 		[[1, 2, 3],
# 		[1, 2, 2],
# 		[-2, -3, -4]
# 		])
# print(euclidean_to_hamming(data, k))

