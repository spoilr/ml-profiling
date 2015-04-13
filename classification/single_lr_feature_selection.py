"""
Logistic Regression Classification
Single LR
Feature selection is applied before
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
from load_data import *
from project_data import *
from parse_theme import *
from split_dataset import *
from feature_selection_cv import *
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from lr_cv import cross_validation

import numpy as np

def feature_selection_before(features, targets, dataset, percentage, ids):
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
		
	known_targets = np.asarray(known_targets)

	cv_features = features_cross_validation(known_dataset, known_targets, features)
	selected_features = select_final_features_from_cv(cv_features, percentage)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)
	known_dataset = sf.extract_data_from_selected_features()

	cross_validation(np.array(known_dataset), known_targets, ids)

	print '####### FEATURES ####### %d \n %s' % (len(selected_features), str(selected_features)) 	

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		feature_selection_before(features, targets, dataset, CV_PERCENTAGE_OCCURENCE_THRESHOLD, ids)
		
	except IndexError:
		print "Error!! Pass 'all' as argument"
