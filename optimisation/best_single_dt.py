"""
Decision Tree Classification
Single DT
Feature selection is applied before
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import *
from parse_theme import *
from feature_selection_before import *
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from cv import dt_one_fold_measures

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		for i in range(1000):
			feature_selection_before(features, targets, dataset, CV_PERCENTAGE_OCCURENCE_THRESHOLD, ids, dt_one_fold_measures, prt=True, file_name="best_single_dt.txt")
		
	except IndexError:
		print "Error!! Pass 'all' as argument"
