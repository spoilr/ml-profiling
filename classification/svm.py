"""
SVM C-Support Vector Classification
Combine SVM for themes
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
from load_data import *
from optimize_parameters import *
from svm_fusion import *

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	std = StandardizedData(targets)
	dataset, targets = std.thematic_split_and_standardize_dataset() 
	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	cross_validation(dataset, targets, fusion_algorithm, ids)

	optimize = raw_input('Optimise parameters? y or n')
	if optimize == 'y':
			score = raw_input('Score? accuracy, recall, precision')
			opt_params = OptimizeParameters(dataset, targets)
			theme_index = int(raw_input('Theme index? 0, 1 or 2'))
			opt_params.category_optimize_parameters(score, theme_index)