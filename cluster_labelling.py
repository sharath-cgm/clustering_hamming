# label the clusters
# Maximum weighted matching problem using Hungarian Algorithm

"""
input: 
cluster_labels - labels after running k-means
true_labels - labels from the dataset
k - number of distinct labels
number_samples - number of datapoints/samples

Returns:
cluster_labels - predicted labels to each datapoint

"""

import dlib
import numpy as np

def labelling(cluster_labels, true_labels, k, number_samples):

    matching = np.array([[0 for x in range(k)] for y in range(k)])
    for i in range(number_samples):
        matching[cluster_labels[i]][true_labels[i]] += 1

    assignment = dlib.max_cost_assignment(dlib.matrix(matching))
    
    # print(assignment)
    # matches = dlib.assignment_cost(dlib.matrix(matching), assignment)
    # print(matches)
    # accuracy = matches/number_samples
    # print(accuracy)

    for i in range(number_samples):
        cluster_labels[i] = assignment[cluster_labels[i]]

    return cluster_labels