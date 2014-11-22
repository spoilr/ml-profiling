"""
===============================
Univariate Feature Selection
===============================

An example showing univariate feature selection.

For each feature, we plot the p-values for the univariate feature selection and the corresponding
weights of an SVM. We can see that univariate feature selection selects the informative features 
and that these have larger SVM weights.

Applying univariate feature selection before the SVM increases the SVM weight attributed to the 
significant features, and will thus improve classification.
"""

print(__doc__)

import sys
sys.path.insert(0, './utils/')
from load_data import *
from parse_theme import *

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

###############################################################################

def univariate_feature_selection(dataset, features):
	# load the dataset
	spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	targets = data.targets


	X = dataset
	y = data.targets


	###############################################################################
	plt.figure(1)
	plt.clf()

	X_indices = np.arange(X.shape[-1])

	###############################################################################
	# Univariate feature selection with F-test for feature scoring
	# We use the default selection function: the 10% most significant features
	selector = SelectPercentile(f_classif, percentile=10)
	selector.fit(X, y)
	scores = -np.log10(selector.pvalues_)
	scores /= scores.max()
	plt.bar(X_indices - .45, scores, width=.2,
	        label=r'Univariate score ($-Log(p_{value})$)', color='g')

	###############################################################################
	# Compare to the weights of an SVM
	clf = svm.SVC(kernel='linear')
	clf.fit(X, y)

	svm_weights = (clf.coef_ ** 2).sum(axis=0)
	svm_weights /= svm_weights.max()

	plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')

	clf_selected = svm.SVC(kernel='linear')
	clf_selected.fit(selector.transform(X), y)

	svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
	svm_weights_selected /= svm_weights_selected.max()

	plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
	        width=.2, label='SVM weights after selection', color='b')


	x = np.arange(0, len(features))
	plt.title("Comparing feature selection")
	plt.xlabel('Feature number')
	plt.xticks(x, features, rotation=45)
	plt.yticks(())
	#plt.axis('tight')
	plt.legend(loc='upper right')
	plt.show()

if __name__ == "__main__":
	[dataset, features] = parse_theme(sys.argv[1])
	univariate_feature_selection(dataset, features)