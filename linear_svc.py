"""
Linear Support Vector Classification.
Similar to SVC with parameter kernel='linear', but implemented in terms of liblinear rather than libsvm, 
so it has more flexibility in the choice of penalties and loss functions and should scale better (to large numbers of samples).
This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.


NOTE: The underlying C implementation uses a random number generator to select features when fitting the model. 
It is thus not uncommon, to have slightly different results for the same input data. If that happens, try with a smaller tol parameter.
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *

import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# splits dataset into 2 sets: one for which we know the target 
# and one for which the target is unknown
def split_dataset(dataset, targets):
	unknowns = []
	known_dataset = []
	known_targets = []
	for i in range(0, len(targets)):
		if targets[i] == 0:
			unknowns.append(dataset[i])
		else:	
			known_dataset.append(dataset[i])
			known_targets.append(targets[i])

	return [np.array(known_dataset), known_targets, np.array(unknowns)]	

def linear_svc(dataset, targets):
	[known_dataset, known_targets, unknowns, ] = split_dataset(dataset, targets)

	model = LinearSVC(tol=0.00001, random_state=None)
	model.fit(known_dataset, known_targets)
	print 'Model score: %f' % model.score(known_dataset, known_targets)


if __name__ == "__main__":
	spreadsheet = Spreadsheet('/home/user/Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme(sys.argv[1])
	linear_svc(dataset, targets)


