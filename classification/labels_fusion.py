"""
Combine outputs of X classifiers.
Each classifier produces a class label.
"""

print(__doc__)

import scipy.misc as misc
import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from split_dataset import *
from collections import Counter

NR_THEMES = 3

# the number of classifiers is odd
# the classifier outputs are assumed to be independent
# accuracies must be ordered (increasing)
def majority_vote(predictions, y_test, accuracies):
	combined_predictions = []
	for i in range(0, len(y_test)):
		data = Counter(predictions[:,i])
		combined_predictions.append(data.most_common(1)[0][0])

	maj_vote_bounds(accuracies)
	return combined_predictions

def upper_epsilons(nr_classifiers, k, accuracies):
	epsilons = []
	epsilons.append(1)
	for m in range (1, k+1):
		eps = 1/float(m) * sum([accuracies[i] for i in range(nr_classifiers-k+m)])
		epsilons.append(eps)
	return min(epsilons)	

def lower_epsilons(nr_classifiers, k, accuracies):
	epsilons = []
	epsilons.append(0)
	for m in range (1, k+1):
		eps = 1/float(m) * sum([accuracies[i] for i in range(k-m, nr_classifiers)]) - ((nr_classifiers-k)/float(m))
		epsilons.append(eps)
	return max(epsilons)	

# Matan's Limits on the Majority Vote Accuracy
# upper and lower bounds of the majority vote accuracy in the case of unequal individual accuracies
def maj_vote_bounds(accuracies):
	list.sort(accuracies)
	k = (NR_THEMES + 1) / 2
	upper_eps = upper_epsilons(NR_THEMES, k, accuracies)
	lower_eps = lower_epsilons(NR_THEMES, k, accuracies)
	print 'Upper bound majority vote accuracy %f' % upper_eps
	print 'Lower bound majority vote accuracy %f' % lower_eps

# Littlestone-Warmuth weighted majority
def	weighted_majority(predictions, y_test):
	epsilon = 0.7
	# weights are initialized with 1
	weights = [1, 1, 1]
	print predictions
	print y_test
	for i in range(0, len(y_test)):
		# compute prediction
		out = round(sum([a*b for a,b in zip(predictions[:,i].tolist(), weights)]) / float(sum(weights)))
		#update weights
		for j in range(len(weights)):
			if predictions[j,i] != y_test[i]:
				weights[j] *= (1-epsilon)

	print weights				
	combined_predictions = []
	for i in range(0, len(y_test)):
		# compute final prediction with updated weights
		out = round(sum([a*b for a,b in zip(predictions[:,i].tolist(), weights)]) / float(sum(weights)))
		combined_predictions.append(out)

	return combined_predictions
