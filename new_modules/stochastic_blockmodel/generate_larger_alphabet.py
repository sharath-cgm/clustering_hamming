import numpy as np

def generate_block(m, n, a, sigma, p):
	rand_flips = np.random.uniform(size = (m, n)) < p

	block = np.random.randint(low = 0, high = sigma, size = (m,n))

	for i in range(m):
		for j in range(n):
			if rand_flips[i][j] == True:
				block[i][j] = a[j]

	return block


def generation(N, D, k, sigma, p):
	# written for k = 2

	# generate bias
	a = np.random.randint(low = 0, high = sigma, size = D, dtype = int)
	print(a)

	n = int(N/2)
	d = int(D/2)
	data = np.random.randint(low = 0, high = sigma, size = (N, D), dtype = int)
	data[0:n, 0:d] = generate_block(n, d, a[0:d], sigma, p)
	data[n:N, d:D] = generate_block(n, d, a[d:D], sigma, p)


	labels = np.zeros(shape= (N, 1), dtype = int)
	labels[n : N] = 1

	data = np.concatenate((data, labels) , axis = 1)

	return data



N, D, k = 50000, 500, 2
# N, D, k = 20, 20, 2
# sigma, p = 5, 0.5
sigma, p = 5, 0.3
# sigma, p = 20, 0.25
# sigma, p = 20, 0.1

data = generation(N, D, k, sigma, p)

# print(data)

filename = "N" + str(N) + "_D" + str(D) + "_k" + str(k) + "_sigma" + str(sigma) +"_p" + str(p)+ "_1"
np.savetxt("large_alphabet" + filename + ".txt", data, fmt = '%u', delimiter=" ")

unique_entries, _ = (np.unique(data, axis=0)).shape
print(unique_entries)