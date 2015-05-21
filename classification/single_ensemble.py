"""
Ensemble Classification using DT, KNN, SVM
Single 
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import *
from parse_theme import *
from standardized_data import *
from split_dataset import *
from cv import cv10_ensemble
from cv import dt
from cv import knn
from svms import svm_all_vars

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

		cv10_ensemble(np.array(known_dataset), np.array(known_targets), known_dataset_scaled, dt, knn, svm_all_vars)
		
	except IndexError:
		print "Error!! Pass 'all' as argument"
