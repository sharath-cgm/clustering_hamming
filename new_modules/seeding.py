import numpy as np

def max_ones(data):
	n_samples, _= data.shape
	l, max = [], -1

	for i in range(n_samples):
		count = np.count_nonzero(data[i])
		if count > max:
			l = []
			l.append(i)
			max = count
		elif count == max:
			l.append(i)

	return data[np.random.choice(l)]


def seeding1(data, n_clusters):
	"""
  	data : data in hamming metric
	n_clusters : number of seeds to choose, 'k'

  	---------

  	Returns centers
	"""

	n_samples, n_features = data.shape
	centers = np.empty((n_clusters, n_features), dtype= data.dtype)

	# Pick first center randomly
	center_id = np.random.randint(0, n_samples)
	centers[0] = data[center_id]

	# Perform XOR operation 
	for i in range(n_samples):
		data[i] = np.bitwise_xor(data[i], centers[0])

	# choose the rest of (k-1) centers
	for i in range(1, n_clusters):
		new = max_ones(data)
		centers[i] = new

		for j in range(n_samples):
			data[j] = np.bitwise_xor(data[j], new, out = data[j], where = data[j].astype(dtype = bool))

	return centers


# def seeding2(data, n_clusters):
	"""
	implementation similar to k-means++, 
	"""


data = np.array([ [1, 0, 0], [1,1, 0], [1, 1, 1],  [1, 0, 0] ])
print(seeding1(data, 2))
# print(data)
# print(max_ones(data))