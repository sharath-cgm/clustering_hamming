# 
import numpy as np
from cluster_labelling import labelling
from performance_measures import performance_measures, inertia_analysis, accuracy_analysis, f1_analysis
from sklearn.metrics import accuracy_score, f1_score
from kmeans_hamming import Kmeans

# synthetic dataset
# dataset = "digit_hamming_2.txt"
# dataset = "digits_gray_coded.txt"
# dataset = "mushroom_hamming_larger_alphabet.txt"
dataset = "connect4_hamming_012.txt"
print(dataset)
data = np.loadtxt(dataset)
data = data.astype(int)
labels = data[:, -1]
data = data[:, 0:-1]
# seed = np.array([[1, 0, 0], [0, 1, 1]])
# seed = np.array([[0, 0, 0], [1, 1, 1]])
# data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
# labels = np.array([0, 0, 0, 1,1,1])

# print(seed.shape)

(n_samples, n_features), n_clusters = data.shape, np.unique(labels).size
# print(f"Hand-written dataset: # digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
print(f"# clusters: {n_clusters}; # samples: {n_samples}; # features {n_features}")


# Applying proposed lloyds with best accuracy

best_accuracy, f1, inertia, pred_labels = None, None, None, None
accuracy_list = []
# print("Based on best accuracy, k-means++ seeding: ")
# print("Based on best accuracy, random seeding: ")
# for i in range(5):
	# print(i)
	
# kmeans = Kmeans(init="random", n_clusters=n_clusters, n_init= 5, max_iter = 30, algorithm = "probabilistic_rounding", tanh_t= 7)
# kmeans = Kmeans(init="random", n_clusters=n_clusters, n_init= 10, max_iter = 5, algorithm = "majority_rounding")
# kmeans = Kmeans(init="random", n_clusters=n_clusters, n_init= 10, max_iter = 30, algorithm = "probabilistic_rounding_large_alphabets", tanh_t = 5)
kmeans = Kmeans(init="random", n_clusters=n_clusters, n_init= 10, max_iter = 20, algorithm = "majority_rounding_large_alphabets")


# kmeans = Kmeans(init="seeding2", n_clusters=n_clusters, n_init=1)
kmeans.fit(data, true_labels = labels) #, input_seed = seed)


	# predicted_labels = labelling(kmeans.labels_, labels, n_clusters, n_samples)
	# accuracy = accuracy_score(labels, predicted_labels)

	# accuracy_list.append(accuracy)
	# if (best_accuracy == None) or (best_accuracy > accuracy):
	# 	best_accuracy = accuracy
	# 	pred_labels = kmeans.labels_
	# 	inertia = kmeans.inertia_
	# 	f1 = f1_score(labels, predicted_labels, average='macro')

accuracy_analysis(kmeans.accuracy_list)
# f1_analysis(kmeans.f1_list)
# print("Corresponding F1 score: ", f1)