"""
SVM C-Support Vector Classification
Single SVM
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import *
from parse_theme import *
from standardized_data import *
from cv import cross_validation
from cv import single_svm_one_fold_measures

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		std = StandardizedData(targets, dataset)
		known_dataset_scaled, known_targets = std.split_and_standardize_dataset()

		cross_validation(known_dataset_scaled, known_targets, ids, single_svm_one_fold_measures)
		
	except IndexError:
		print "Error!! Pass 'all' as argument"




			