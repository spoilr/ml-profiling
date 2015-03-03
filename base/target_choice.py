"""
Forensic Target choice 
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
from load_data import *
from split_dataset import *
from parse_theme import *

import numpy as np


# highval = 1
# civil = 2

def feature_occurences(known_dataset, highval, civil, features):
	for i, feature in enumerate(features):
		data_for_feature = known_dataset[:,i]
		highval_data_for_feature = highval[:,i]
		civil_data_for_feature = civil[:,i]
		
		vals = set(data_for_feature)

		with open("features.txt", "a") as myfile:
			myfile.write('\n##############################\n')

		with open("features.txt", "a") as myfile:
			myfile.write('FEATURE %s\n' % feature)
		
		for x in vals:
			highval_occurences = len(np.where(highval_data_for_feature == x)[0])
			civil_occurences = len(np.where(civil_data_for_feature == x)[0])
			
			with open("features.txt", "a") as myfile:
				myfile.write('%d highval %d vs civil %d\n' % (x, highval_occurences, civil_occurences))

def feature_occurences_percentage(known_dataset, highval, civil, features):
	for i, feature in enumerate(features):
		data_for_feature = known_dataset[:,i]
		highval_data_for_feature = highval[:,i]
		civil_data_for_feature = civil[:,i]
		
		vals = set(data_for_feature)

		with open("features_percentage.txt", "a") as myfile:
			myfile.write('\n##############################\n')

		with open("features_percentage.txt", "a") as myfile:
			myfile.write('FEATURE %s\n' % feature)
		


		for x in vals:
			highval_percentage = float(len(np.where(highval_data_for_feature == x)[0])) / len(highval_data_for_feature)
			civil_percentage = float(len(np.where(civil_data_for_feature == x)[0])) / len(civil_data_for_feature)
			
			with open("features_percentage.txt", "a") as myfile:
				myfile.write('%d highval %f vs civil %f\n' % (x, highval_percentage, civil_percentage))


if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = np.array(data.targets)
	ids = data.ids

	[dataset, features] = parse_theme('all')
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	known_targets = np.array(known_targets)

	highval_indices = np.where(known_targets == 1)[0]
	civil_indices = np.where(known_targets == 2)[0]

	highval_data = known_dataset[highval_indices]
	civil_data = known_dataset[civil_indices]

	assert len(highval_indices) == len(highval_data)
	assert len(civil_indices) == len(civil_data)

	print len(highval_indices)
	print len(civil_indices)

	# feature_occurences(known_dataset, highval_data, civil_data, features)
	feature_occurences_percentage(known_dataset, highval_data, civil_data, features)

