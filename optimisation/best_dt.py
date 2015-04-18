"""
Decision Tree Classification
Combine DT for themes
Feature selection is applied before
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import *
from fusion import cross_validation
from fusion import dt
from thematic_data_combined import combine_data_from_feature_selection
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	combined_dataset, targets = combine_data_from_feature_selection(targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD)

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")

	for i in range(1000):
		cross_validation(combined_dataset, targets, fusion_algorithm, ids, dt, prt=True, file_name="best_dt.txt")
