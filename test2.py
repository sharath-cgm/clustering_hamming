# testing using implemented Kmeans

# load data
import numpy as np
from sklearn.datasets import load_digits
from cluster_labelling import labelling
from performance_measures import performance_measures, inertia_analysis, accuracy_analysis
from sklearn.metrics import accuracy_score, f1_score
from cluster_labelling import labelling
from kmeans import Kmeans

data, labels = load_digits(return_X_y=True)
x_squared_norms = (data * data).sum(axis=1) # comment this


(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

# ---------------------------------------------------------------
# Applying Kmeans with best inertia
# print("Based on best inertia, random seeding: ")
# best_inertia, pred_labels = None, None
# inertia_list = []

# for _ in range(50):
# 	kmeans = Kmeans(init="random", n_clusters=n_digits, n_init=1)
# 	kmeans.fit(data)

# 	inertia_list.append(kmeans.inertia_)
# 	if (best_inertia == None) or (best_inertia > kmeans.inertia_):
# 		best_inertia = kmeans.inertia_
# 		pred_labels = kmeans.labels_

# inertia_analysis(inertia_list)
# predicted_labels = labelling(kmeans.labels_, labels, n_digits, n_samples)
# performance_measures(labels, predicted_labels)
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Applying Kmeans with best accuracy

best_accuracy, f1, inertia, pred_labels = None, None, None, None
accuracy_list = []
# print("Based on best accuracy, k-means++ seeding: ")
print("Based on best accuracy, random seeding: ")
for _ in range(50):
	kmeans = Kmeans(init="k-means++", n_clusters=n_digits, n_init=1)
	kmeans.fit(data)

	predicted_labels = labelling(kmeans.labels_, labels, n_digits, n_samples)
	accuracy = accuracy_score(labels, predicted_labels)

	accuracy_list.append(accuracy)
	if (best_accuracy == None) or (best_accuracy > accuracy):
		best_accuracy = accuracy
		pred_labels = kmeans.labels_
		inertia = kmeans.inertia_
		f1 = f1_score(labels, predicted_labels, average='macro')

accuracy_analysis(accuracy_list)
print("Corresponding F1 score: ", f1)
# performance_measures(labels, predicted_labels)
print("Corresponding Inertia: ", inertia)
# ---------------------------------------------------------------



