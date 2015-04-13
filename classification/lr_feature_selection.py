"""
Logistic Regression Classification
Combine LR for themes
Feature selection is applied before
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
from load_data import *
from project_data import *
from lr_fusion import cross_validation
from thematic_data_combined import combine_data_from_feature_selection
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	combined_dataset, targets = combine_data_from_feature_selection(targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD)

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	cross_validation(combined_dataset, targets, fusion_algorithm, ids)

	