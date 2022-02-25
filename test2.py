# testing using implemented Kmeans

# load data
import numpy as np
from sklearn.datasets import load_digits

data, labels = load_digits(return_X_y=True)

(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")



from cluster_labelling import labelling
from performance_measures import performance_measures
from kmeans import Kmeans, Temp
from cluster_labelling import labelling
from performance_measures import performance_measures

for _ in range(5):

	kmeans = Kmeans(init="k-means++", n_clusters=n_digits, n_init=10, random_state=None)
	# kmeans.test_temp()
	# kmeans.fit(data)
	x_squared_norms = (data * data).sum(axis=1)
	c = kmeans._init_centroids(X = data, x_squared_norms = x_squared_norms, init = "random", random_state = None)
	print(c)

    # predicted_labels = labelling(kmeans.labels_, labels, n_digits, n_samples)
    # performance_measures(labels, predicted_labels)