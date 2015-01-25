print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from split_dataset import *
from project_data import *

import operator
import numpy as np
import matplotlib.pyplot as plt

# A statistical property, called information gain, is used. 
# Gain measures how well a given attribute separates training examples into targeted classes. 
# in DT The one with the highest information (information being the most useful for classification) is selected. 
# Entropy measures the amount of information in an attribute.
def entropy(targets):
	length = len(targets)
	highval = float(len([x for x in targets if x == 1])) # float needed for division
	civil = float(length - highval) # float needed for division

	if highval == civil:
		entropy = 1
	elif highval == length or civil == length:
		entropy = 0
	else:
		highvalprob = highval/length
		civilprob = civil/length

		log2targets = np.log2(np.array([highvalprob, civilprob]))
		entropy = -highvalprob * log2targets[0] - civilprob * log2targets[1]

	return entropy

# Information gain IG(A) is the measure of the difference in entropy from before to after the set S is split on an attribute A. 
# ie How much uncertainty in S was reduced after splitting set S on attribute A.
def attribute_info_gain(targets, attribute_vals, entropy):
	length = len(targets)
	values = set(attribute_vals.tolist())

	entropies = sum([prob_entropy_for_value(val, attribute_vals, targets) for val in values])
	return entropy - entropies

def info_gain(dataset, features, targets, sys_entropy):
	info_gains = []
	nr_columns = dataset.shape[1]
	for i in range(0, nr_columns):
		attr_gain = attribute_info_gain(targets, dataset[:,i], sys_entropy)
		info_gains.append(attr_gain)
		#print "%s info gain %f" % (features[i], attr_gain)
	return info_gains	

def prob_entropy_for_value(value, attribute_vals, targets):
	targets_for_value = [targets[i] for i in range(0, len(targets)) if attribute_vals[i] == value]
	return entropy(targets_for_value) * float(len(targets_for_value)) / len(targets)

def split_info(attribute_vals):
	values = set(attribute_vals.tolist())
	prob_for_values = []
	attribute_vals_list = attribute_vals.tolist()
	for x in values:
		prob_for_values.append(float(attribute_vals_list.count(x)) / len(attribute_vals_list)) 
	log2probs = np.log2(np.array(prob_for_values))	
	zipped = zip(prob_for_values, log2probs)
	sum = 0
	for (x,y) in zipped:
		sum += x * y
	return -sum	

def gain_ratio_for_val(targets, attribute_vals, entropy):
	info_gain = attribute_info_gain(targets, attribute_vals, entropy)
	attr_split_info = split_info(attribute_vals)
	return info_gain / attr_split_info

def gain_ratio(dataset, features, targets, sys_entropy):
	gain_ratios = []
	nr_columns = dataset.shape[1]
	for i in range(0, nr_columns):
		attr_gain_ratio = gain_ratio_for_val(targets, dataset[:,i], sys_entropy)
		gain_ratios.append(attr_gain_ratio)
		#print "%s gain ratio %f" % (features[i], attr_gain_ratio)
	return gain_ratios	

# tennis example
def test_function():
	features = ["out", "temp", "humid", "wind"]
	targets = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]
	dataset = np.array([[1,1,1,1], [1,1,1,2], [2,1,1,1], [3,3,1,1], [3,2,2,1], [3,2,2,2,], [2,2,2,2], [1,3,1,1], [1,2,2,1], [3,3,2,1], [1,3,2,2], [2,3,1,2], [2,1,2,1], [3,3,1,2]])
	assert len(dataset) == 14
	sys_entropy = entropy(targets)
	print "entropy of system %f" % sys_entropy
	print "############ INFO GAIN ############"
	info_gain(dataset, features, targets, sys_entropy)
	print "############ GAIN RATIO ############"
	gain_ratio(dataset, features, targets, sys_entropy)

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme(sys.argv[1])
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)

	sys_entropy = entropy(known_targets)
	print "entropy of system %f" % sys_entropy

	print "\n\n\n############ INFO GAIN ############"
	info_gain(dataset, features, targets, sys_entropy)
	print "\n\n\n############ GAIN RATIO ############"
	gain_ratio(dataset, features, targets, sys_entropy)

	print '\n\n\n######### TEST ###########'
	test_function()


