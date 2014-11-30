print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from split_dataset import *
import itertools
from feature_entropy import *

import operator
import numpy as np
import matplotlib.pyplot as plt

def feature_context(dataset, targets, features):
	sys_entropy = entropy(targets)
	feature_combinations_indexes = combinations_of_join_features(dataset)
	for (index1, index2) in feature_combinations_indexes:
		attr1 = dataset[:, index1]
		attr2 = dataset[:, index2]
		res = check_derived_feature_context(attr1, attr2, sys_entropy, targets)
		if res is not None:
			print "%s - %s provides context" % (features[index1], features[index2])
		else:
			print "%s - %s does NOT provide context" % (features[index1], features[index2])	


# A derived feature is a candidate for feature selection if 
# its correlation with the class is higher than both of its constituent features.
def check_derived_feature_context(attr1, attr2, sys_entropy, targets):
	possible_joined_values = all_possible_values_after_join(attr1, attr2)
	derived_feature_values = join_features(attr1, attr2, possible_joined_values)
	derived_feature_gain = attribute_info_gain(targets, derived_feature_values, sys_entropy)
	attr1_gain = attribute_info_gain(targets, attr1, sys_entropy)
	attr2_gain = attribute_info_gain(targets, attr2, sys_entropy)
	if derived_feature_gain > attr1_gain and derived_feature_gain > attr2_gain:
		print "%f - %f and %f" % (derived_feature_gain, attr1_gain, attr2_gain)
		return (attr1, attr2)

def join_features(attr1, attr2, possible_joined_values):
	assert attr1.shape[0] == attr2.shape[0]
	derived_feature_values = []
	for i in range(0, attr1.shape[0]):
		# find index of combined pair and make this the new value
		derived_feature_values.append(possible_joined_values.index((attr1[i], attr2[i])))
	#return np.asarray(derived_feature_values).reshape(-1, 1) # column vector
	return np.asarray(derived_feature_values)

def combinations_of_join_features(dataset):
	nr_columns = dataset.shape[1]
	indexes = list(xrange(nr_columns))
	combinations_of_join_features = []
	for x in itertools.combinations(indexes, 2):
		combinations_of_join_features.append(x)
	return combinations_of_join_features	

def all_possible_values_after_join(attr1, attr2):
	iterables = [set(attr1.tolist()), set(attr2.tolist())]
	combinations = []
	for x in itertools.product(*iterables):
		combinations.append(x)
	return combinations	

# tennis example
def test_function():
	features = ["out", "temp", "humid", "wind"]
	targets = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]
	dataset = np.array([[1,1,1,1], [1,1,1,2], [2,1,1,1], [3,3,1,1], [3,2,2,1], [3,2,2,2,], [2,2,2,2], [1,3,1,1], [1,2,2,1], [3,3,2,1], [1,3,2,2], [2,3,1,2], [2,1,2,1], [3,3,1,2]])
	assert len(dataset) == 14
	sys_entropy = entropy(targets)
	print "entropy of system %f" % sys_entropy
	info_gain(dataset, features, targets, sys_entropy)
	feature_context(dataset, targets, features)

if __name__ == "__main__":
	spreadsheet = Spreadsheet('/home/user/Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme(sys.argv[1])
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)

	#feature_context(known_dataset, known_targets, features)

	test_function()