from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from cluster_labelling import labelling

def performance_measures(true_labels, predicted_labels, verbose = False):

	if verbose is True:
		print(
			f"Report: \n"
			f"{metrics.classification_report(true_labels, predicted_labels)}\n"
		)
	else:
		accuracy = accuracy_score(true_labels, predicted_labels)
		f1 = f1_score(true_labels, predicted_labels, average='macro')
		print(
			f"Accuracy: {accuracy}, F1-score: {f1}"
			)

	return None

def accuracy_analysis(accuracy_list):
	average = np.average(accuracy_list)
	standard_deviation = np.std(accuracy_list)
	maximum = max(accuracy_list)

	print("Accuracy Analysis: Average = ", average, " ; Standard Deviation = ", standard_deviation, "; Best Accuracy = ", maximum)

	return None

def inertia_analysis(inertia_list):
	average = np.average(inertia_list)
	standard_deviation = np.std(inertia_list)
	maximum = min(inertia_list)

	print("Inertia Analysis: Average = ", average, " ; Standard Deviation = ", standard_deviation, "; Best Inertia = ", maximum)

	return None