# load data
import numpy as np
from sklearn.datasets import load_digits

data, labels = load_digits(return_X_y=True)

(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")


# TODO: this translation should be in the individual functions
# subtract the mean of x for more accurate distance computations
# data_mean = data.mean(axis=0)
# data -= data_mean

""" ---------------------------------------------------------------- """

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from time import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cluster_labelling import labelling
from performance_measures import performance_measures

# print(82 * "_")
# print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

for _ in range(5):
    # t0 = time()
    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=5, random_state=None)
    # kmeans.fit(data)
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    # fit_time = time() - t0

    # bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels, estimator = estimator, fit_time=fit_time)
    ## print(kmeans)

    predicted_labels = labelling(kmeans.labels_, labels, n_digits, n_samples)
    performance_measures(labels, predicted_labels)


# for _ in range(5):
#   t0 = time()
#   kmeans = KMeans(init="random", n_clusters=n_digits, n_init=1, random_state=None)
#   estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
#   fit_time = time() - t0

#   # bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels, estimator = estimator, fit_time=fit_time)
#   labelling(kmeans.labels_)

# print(82 * "_")