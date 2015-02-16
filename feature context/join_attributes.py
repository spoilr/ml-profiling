print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from split_dataset import *
import itertools
from feature_entropy import *
from project_data import *
from parameters import TOP_FEATURES_PERCENTAGE_THRESHOLD

import math as math
import operator
import numpy as np
import matplotlib.pyplot as plt

def feature_context(dataset, targets, features):
	sys_entropy = entropy(targets)
	feature_combinations_indexes = combinations_of_join_features(dataset)
	feature_gain_ratio = get_feature_gain_ratio(dataset, targets, features, sys_entropy)

	avg_gain_ratios = np.mean(np.array(gain_ratio(dataset, features, targets, sys_entropy)))
	orig_feats = set(features)
	feats = set()
	der_feats = dict()

	for (index1, index2) in feature_combinations_indexes:
		attr1 = dataset[:, index1]
		attr2 = dataset[:, index2]
		attr1_gain = feature_gain_ratio[index1]
		attr2_gain = feature_gain_ratio[index2]
		selected_derived_feature_gain = check_derived_feature_context(attr1, attr2, attr1_gain, attr2_gain, sys_entropy, targets)
		if selected_derived_feature_gain is not None:
			# print "der %f : %s with %f - %s with %f provides context" % (selected_derived_feature_gain, features[index1], attr1_gain, features[index2], attr2_gain)
			feats.add(features[index1])
			feats.add(features[index2])
			der_feats[(index1, index2)] = selected_derived_feature_gain
		# else:
		# 	print "%s - %s does NOT provide context" % (features[index1], features[index2])	

	#### for testing 	
	# print 'FEATURE GAIN RATIO %s' % str(feature_gain_ratio)
	print 'AVG GAIN RATIO %f' % avg_gain_ratios
	print 'selected %d vs %d that are above avg' % (len([x for x in feature_gain_ratio if x >= avg_gain_ratios]), len(feature_gain_ratio))
	
	#return feats
	return final_set_of_features(features, feature_gain_ratio, avg_gain_ratios, der_feats)

def final_set_of_features(features, feature_gain_ratio, avg_gain_ratios, der_feats):
	selected_features_from_whole_set = [features[i] for i in range(len(feature_gain_ratio)) if feature_gain_ratio[i] >= avg_gain_ratios]
	selected_features_from_derived = []

	sorted_der_feats = sorted(der_feats.items(), key=operator.itemgetter(1))
	nr_times = int(math.floor(TOP_FEATURES_PERCENTAGE_THRESHOLD * len(der_feats)))
	
	for i in range(nr_times):
		selected_features_from_derived.append(features[sorted_der_feats[i][0][0]])
		selected_features_from_derived.append(features[sorted_der_feats[i][0][1]])

	#return set(selected_features_from_whole_set).union(selected_features_from_derived)	
	return set(selected_features_from_derived)

def get_feature_gain_ratio(dataset, targets, features, sys_entropy):
	# check that dataset has the same nr of features as features
	assert dataset.shape[1] == len(features)
	feature_gain_ratio = []
	for i in range(len(features)):
		attr = dataset[:, i]
		feat_gain_ratio = gain_ratio_for_val(targets, attr, sys_entropy)
		feature_gain_ratio.append(feat_gain_ratio)
	return feature_gain_ratio


# A derived feature is a candidate for feature selection if 
# its correlation with the class is higher than both of its constituent features.
def check_derived_feature_context(attr1, attr2, attr1_gain, attr2_gain, sys_entropy, targets):
	possible_joined_values = all_possible_values_after_join(attr1, attr2)
	derived_feature_values = join_features(attr1, attr2, possible_joined_values)
	derived_feature_gain = gain_ratio_for_val(targets, derived_feature_values, sys_entropy)
	if derived_feature_gain > attr1_gain and derived_feature_gain > attr2_gain:
		#print "%f - %f and %f" % (derived_feature_gain, attr1_gain, attr2_gain)
		return derived_feature_gain

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
	print "SYSTEM ENTROPY %f" % sys_entropy
	print "############ INFO GAIN ############"
	info_gain(dataset, features, targets, sys_entropy)
	print "############ GAIN RATIO ############"
	gain_ratio(dataset, features, targets, sys_entropy)
	print "############ CONTEXT ############"
	feats = feature_context(dataset, targets, features)

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme(sys.argv[1])
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)

	feats = feature_context(known_dataset, known_targets, features)

	#test_function()