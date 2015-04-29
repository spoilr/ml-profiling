"""
SVM C-Support Vector Classification
Combine SVM for themes
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import *
from fusion import cv10
from svms import svm_for_features_fusion
from standardized_data import *

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	std = StandardizedData(targets)
	dataset, targets = std.thematic_split_and_standardize_dataset() 

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	cv10(dataset, targets, fusion_algorithm, ids, svm_for_features_fusion)

	