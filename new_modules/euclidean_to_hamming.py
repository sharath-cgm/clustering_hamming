"""
main functions:
1) euclidean_to_hamming(X, k): X- data in euclidean metric, k - no of clusters
	|- hyperparameter- no_trials
2) check_repetitions(X): X- data in hamming metric, out (optional): to print the number of samples and unique samples

"""

"""
todo: 
1) generate random matrix of appropriate dimension and use fast matrix multiplication function: this could be faster

"""


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
		# generate D-dimensional 'hamming_dim' no. of random vectors (each element chosen uniformly at random in range [-1, 1) ). -> this doesnt work
		# rand_normal = np.random.uniform(-1, 1, D)
		# generate D-dimensional 'hamming_dim' no. of random vectors with gaussian ditribution
		rand_normal = np.random.normal(loc = 0, scale = 1, size = D)


		for j in range(N):
			# take inner product, if positive- 1, else 0
			product = np.dot(X[j], rand_normal)
			if product > 0:
				hamming_data[j][i] = 1

	# rand_normal_matrix = np.random.normal(loc = 0, scale = 1, size = ())
	# hamming_data = multiple data * rand_normal_matrix
	# signum(data)

	return hamming_data

def euclidean_to_hamming(X, k):
	no_trials = 15
	for _ in range(no_trials):
		# hamming_data = convert(X, k+1) # hamming_dim = k+1

		_, D = X.shape
		hamming_data = convert(X, D) # hamming_dim = D
		
		if check_repetitions(hamming_data):
			break

	return hamming_data


# data = np.array(
# 		[[1, 2, 3],
# 		[1, 2, 2],
# 		[-2, -3, -4]
# 		])
# print(euclidean_to_hamming(data, 2))
