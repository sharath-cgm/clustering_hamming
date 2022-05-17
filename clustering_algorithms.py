from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

import numpy as np
from cluster_labelling import labelling
from performance_measures import performance_measures, inertia_analysis, accuracy_analysis, f1_analysis
from cluster_labelling import labelling
from sklearn.metrics import accuracy_score, f1_score

# load data
# synthetic dataset
dataset_name = "N100_D20_k2_1.txt"
print(dataset_name)
dataset = np.loadtxt(dataset_name)
labels = dataset[:, -1]
labels = labels.astype(int)
data = dataset[:, 0:-1]

(n_samples, n_features), n_clusters = data.shape, np.unique(labels).size

print(f"# classes: {n_clusters}; # samples: {n_samples}; # features {n_features}")


######## DBScan
print("\nResults of DBSCAN:\n")
best_accuracy, best_f1 = None, None
accuracy_list, f1_list = [], []

no_epochs= 1

for i in range(no_epochs):
	# shuffle data
	# np.random.shuffle(dataset)
	# labels = dataset[:, -1]
	# labels = labels.astype(int)
	# data = dataset[:, 0:-1]

	# run algorithm
	db = DBSCAN(eps = 2.5, min_samples = 2* n_features).fit(data)

	# print(db.labels_)
	n_clusters_estimated = len(set(db.labels_)) - (1 if -1 in labels else 0)
	n_noise_ = list(db.labels_).count(-1)

	print("Estimated number of clusters: %d" % n_clusters_estimated)
	print("Estimated number of noise points: %d" % n_noise_)

	# analysis
	predicted_labels = labelling(db.labels_, labels, n_clusters, n_samples)

	accuracy = accuracy_score(labels, predicted_labels)
	# print(predicted_labels)

	accuracy_list.append(accuracy)

	f1 = f1_score(labels, predicted_labels, average='macro')
	f1_list.append(f1)



accuracy_analysis(accuracy_list)
f1_analysis(f1_list)

print("\n\n")


#### BIRCH
print("Results of BIRCH\n")

best_accuracy, best_f1 = None, None
accuracy_list, f1_list = [], []

no_epochs= 1

for i in range(no_epochs):
	# shuffle data
	# np.random.shuffle(dataset)
	# labels = dataset[:, -1]
	# labels = labels.astype(int)
	# data = dataset[:, 0:-1]

	# run algorithm
	brc = Birch(n_clusters = n_clusters).fit(data)

	# print(brc.labels_)

	# analysis
	predicted_labels = labelling(brc.labels_, labels, n_clusters, n_samples)

	accuracy = accuracy_score(labels, predicted_labels)
	# print(predicted_labels)

	accuracy_list.append(accuracy)

	f1 = f1_score(labels, predicted_labels, average='macro')
	f1_list.append(f1)



accuracy_analysis(accuracy_list)
f1_analysis(f1_list)

print("\n\n")

######### hierarchical clustering

print("Results of hierarchical clustering\n")
best_accuracy, best_f1 = None, None
accuracy_list, f1_list = [], []

no_epochs= 5

for i in range(no_epochs):
	# shuffle data
	np.random.shuffle(dataset)
	labels = dataset[:, -1]
	labels = labels.astype(int)
	data = dataset[:, 0:-1]
	
	# run algorithm
	clustering = AgglomerativeClustering(n_clusters = n_clusters, linkage='ward').fit(data)

	# analysis
	predicted_labels = labelling(clustering.labels_, labels, n_clusters, n_samples)

	accuracy = accuracy_score(labels, predicted_labels)
	# print(predicted_labels)

	accuracy_list.append(accuracy)

	f1 = f1_score(labels, predicted_labels, average='macro')
	f1_list.append(f1)


accuracy_analysis(accuracy_list)
f1_analysis(f1_list)

print("\n\n")


########### lloyds with kmeans++ seeding
print("Results of Lloyds with k-means++ seeding\n")

best_accuracy, best_f1 = None, None
accuracy_list, f1_list = [], []

no_epochs= 5
for _ in range(no_epochs):
	# shuffle data
	# np.random.shuffle(dataset)
	# labels = dataset[:, -1]
	# labels = labels.astype(int)
	# data = dataset[:, 0:-1]

	kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=1,random_state=None)
	kmeans.fit(data)

	# analysis
	predicted_labels = labelling(clustering.labels_, labels, n_clusters, n_samples)

	accuracy = accuracy_score(labels, predicted_labels)
	# print(predicted_labels)

	accuracy_list.append(accuracy)

	f1 = f1_score(labels, predicted_labels, average='macro')
	f1_list.append(f1)


accuracy_analysis(accuracy_list)
f1_analysis(f1_list)