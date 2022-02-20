"""
Performance measure used: Accuracy, f1-score

"""

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score

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


	return 0 