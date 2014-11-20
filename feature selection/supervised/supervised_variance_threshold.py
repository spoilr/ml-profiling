"""
================================
Supervised Variance Threshold
================================
"""

''' Feature variance for each class.'''

print(__doc__)

import sys
sys.path.insert(0, './utils/')
from load_data import *

import operator
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.utils import validation


def extract_target_class_examples(target, dataset):
	sample = [];
	for i in range(0, len(targets)):
		if targets[i] == target:
			sample.append(dataset[i])
	return np.array(sample)

def target_class_variance(target_class, features):
	feature_variance = np.var(target_class, axis = 0)
	assert len(feature_variance) == len(target_class[0])

	# create dictionary of feature and variance
	variance_per_feature = {}
	for i in range(0, len(feature_variance)):
		variance_per_feature[features[i]] = feature_variance[i]

	decreasing_variance_per_feature = OrderedDict(sorted(variance_per_feature.items(), key=lambda t: t[1], reverse = True))

	# create tuples of feature and variance
	variances = sorted(variance_per_feature.items(), key=operator.itemgetter(1), reverse=True)

	#######################################################
	# plot feature variance
	N = len(variances)
	x = np.arange(1, N+1)
	y = [num for (s, num) in variances]
	labels = [s for (s, num) in variances]
	width = 1
	bar1 = plt.bar(x, y, width, color="y")
	plt.ylabel('Variance')
	plt.xticks(x + width/2.0, labels, rotation=45)
	plt.gcf().subplots_adjust(bottom=0.25)
	plt.show()


# load the dataset
spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
data = Data(spreadsheet)
targets = data.targets

print len(targets)
print len([x for x in targets if x == 0])
print len([x for x in targets if x == 1])
print len([x for x in targets if x == 2])


dataset = data.extract_illness_examples()
features = data.illness_features


highval = extract_target_class_examples(1, dataset)
civil = extract_target_class_examples(2, dataset)

target_class_variance(highval, features)
target_class_variance(civil, features)

