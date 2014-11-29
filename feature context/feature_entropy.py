print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *

import operator
import numpy as np
import matplotlib.pyplot as plt

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
def info_gain(dataset, targets, attribute_col, entropy):
	length = len(targets)
	values = set(attribute_col.tolist())

	entropies = sum([prob_entropy_for_value(val, attribute_col, targets) for val in values])
	return entropy - entropies

def prob_entropy_for_value(value, attribute_col, targets):
	targets_for_value = [targets[i] for i in range(0, len(targets)) if attribute_col[i] == value]
	return entropy(targets_for_value) * float(len(targets_for_value)) / len(targets)

# tennis example
def test_function():
	targets = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]
	dataset = np.array([[1,1,1,1], [1,1,1,2], [2,1,1,1], [3,3,1,1], [3,2,2,1], [3,2,2,2,], [2,2,2,2], [1,3,1,1], [1,2,2,1], [3,3,2,1], [1,3,2,2], [2,3,1,2], [2,1,2,1], [3,3,1,2]])
	assert len(dataset) == 14
	sys_entropy = entropy(targets)
	out_gain = info_gain(dataset, targets, dataset[:,0], sys_entropy)
	temp_gain = info_gain(dataset, targets, dataset[:,1], sys_entropy)
	hum_gain = info_gain(dataset, targets, dataset[:,2], sys_entropy)
	wind_gain = info_gain(dataset, targets, dataset[:,3], sys_entropy)
	print "entropy %f" % sys_entropy
	print "out %f" % out_gain
	print "temp %f" % temp_gain
	print "hum %f" % hum_gain
	print "wind %f" % wind_gain

if __name__ == "__main__":
	spreadsheet = Spreadsheet('/home/user/Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme(sys.argv[1])
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	entropy(known_targets)

	# test_function()


