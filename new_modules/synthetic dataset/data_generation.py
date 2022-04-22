from sklearn.metrics.pairwise import manhattan_distances
import numpy as np

def quality_data(centers):
	# measure min hamming distance between all pairs of centers

	n_centers, min_dist = centers.shape

	for i in range(n_centers):
		for j in range(i+1, n_centers):
			dist = manhattan_distances([centers[i]],[centers[j]])[0][0]
			if min_dist > dist:
				min_dist = dist

	return min_dist

def generation(N, D, k, alpha = 0.2):
	"""
	Input:
	N : number of samples to generated
	D : dimension
	k : number of distinct clusters
	alpha : cluster clarity (measure of how compact each cluster is)
	------------------------------------
	returns-
	data: (N * D) array
	centers: (k * D) array
	labels: (D*1) array
	"""

	# randomly choose k D-dimensional centers
	centers= np.random.randint(2, size= (k, D))
	labels = np.array([list(range(k))])

	data = np.array(centers, dtype = int)
	# Add N/k - 1 points into each cluster
	n_samples = int(N/k - 1)
	for i in range(k):
		rand_flips = np.random.uniform(size = (n_samples, D)) < alpha
		samples_cluster = np.tile(centers[i], (n_samples, 1))
		np.logical_not(samples_cluster, out = samples_cluster, where = rand_flips)
		data = np.concatenate((data, samples_cluster), axis = 0)

		label_tiled = np.tile([i], (1, n_samples))
		labels = np.concatenate((labels, label_tiled), axis = 1)

	data = np.concatenate((data, labels.transpose()) , axis = 1)

	return data, centers

N, D, k = 10000, 100, 20
# data, centers, labels = generation(N = 100000, D = 1000, k = 10000)
# data, centers, labels = generation(N = 100000, D = 1000, k = 2)
# data, centers, labels = generation(N = 500000, D = 50, k = 2)
# data, centers, labels = generation(N = 500000, D = 50, k = 20000)
data, centers = generation(N, D, k)


# print(type(data[0][0]))
# data, centers, labels = generation(N = 50000, D = 50, k = 100)
# print(data.shape)
# print("\n\n",centers)
# np.savetxt("N_3000_2.txt", np.append(data, labels, axis = 1), delimiter=" ")
# print("writing")
# np.savetxt("N_10000_1.csv", data, delimiter=" ")
filename = "N" + str(N) + "_D" + str(D) + "_k" + str(k) + "_1"
np.savetxt(filename + ".txt", data, fmt = '%u', delimiter=" ")


unique_entries, _ = (np.unique(data, axis=0)).shape
print(unique_entries)

print(quality_data(centers))
