"""
Ensemble Classification using DT, KNN, SVM
Single Feature selection
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
from load_data import *
from project_data import *
from parse_theme import *
from feature_selection_before import *
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from cv import dt
from cv import knn
from svms import svm_selected_vars

import numpy as np

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		[known_dataset, known_targets, unk] = split_dataset(dataset, targets)

		[dataset, features] = parse_theme(sys.argv[1])
		std = StandardizedData(targets, dataset)
		known_dataset_scaled, known_targets = std.split_and_standardize_dataset()

		for i in range(100):
			feature_selection_before_ensemble(features, targets, dataset, CV_PERCENTAGE_OCCURENCE_THRESHOLD, dt, knn, svm_selected_vars, prt=True, file_name="single_ensemble_ft.txt")
		
	except IndexError:
		print "Error!! Pass 'all' as argument"

