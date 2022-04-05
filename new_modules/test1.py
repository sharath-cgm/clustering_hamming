from euclidean_to_hamming import euclidean_to_hamming, check_repetitions

# load data
import numpy as np
from sklearn.datasets import load_digits

data, labels = load_digits(return_X_y=True)

print(type(labels))

(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

data = euclidean_to_hamming(data, n_digits)

labels = (np.array([labels])).transpose()

check_repetitions(data, out = True)
print(data[0])
np.savetxt("digit_hamming.txt", np.append(data, labels, axis = 1), delimiter= " ")