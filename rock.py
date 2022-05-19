from pyclustering.cluster.rock import rock

import numpy as np
from cluster_labelling import labelling
from performance_measures import performance_measures, inertia_analysis, accuracy_analysis, f1_analysis
from cluster_labelling import labelling
from sklearn.metrics import accuracy_score, f1_score

# load data
# synthetic dataset
dataset_name = "mushroom_hamming.txt"
print(dataset_name)
dataset = np.loadtxt(dataset_name)
labels = dataset[:, -1]
labels = labels.astype(int)
data = dataset[:, 0:-1]

(n_samples, n_features), n_clusters = data.shape, np.unique(labels).size

print(f"# classes: {n_clusters}; # samples: {n_samples}; # features {n_features}")


best_accuracy, best_f1 = None, None
accuracy_list, f1_list = [], []

# rock
no_epochs= 5
for _ in range(no_epochs):

	rock_instance = rock(data = data, eps = 2.5, number_clusters = n_clusters)
	rock_instance.process()

	clusters = rock_instance.get_clusters()
	# print(clusters)
	rock_labels = np.zeros(n_samples)
	for i in range(len(clusters)):
		# print(clusters[i])
		for j in clusters[i]:
			rock_labels[j] = i

	# print(rock_labels)

	predicted_labels = labelling(rock_labels, labels, n_clusters, n_samples)

	accuracy = accuracy_score(labels, predicted_labels)
	# print(predicted_labels)

	accuracy_list.append(accuracy)

	f1 = f1_score(labels, predicted_labels, average='macro')
	f1_list.append(f1)


accuracy_analysis(accuracy_list)
f1_analysis(f1_list)


