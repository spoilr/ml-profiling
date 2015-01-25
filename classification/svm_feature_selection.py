"""
SVM C-Support Vector Classification
Combine SVM for themes
Feature selection is applied before
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
from load_data import *
from optimize_parameters import *
from svm_fusion import *
from selected_features import *
sys.path.insert(0, 'feature context/')
from feature_selection_cv import *

themes = ['net', 'ill', 'ideo']

def thematic_data_from_feature_selection(theme, percentage):
	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	
	known_targets = np.asarray(known_targets)

	cv_features = features_cross_validation(known_dataset, known_targets, features)
	selected_features = select_final_features_from_cv(cv_features, percentage)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)

	print '####### FEATURES ####### %d %s' % (len(selected_features), str(selected_features)) 

	return sf.extract_data_from_selected_features(), known_targets

def combine_data_from_feature_selection(percentage):
	combined_dataset = []
	targets = []
	for theme in themes:
		data, targets = thematic_data_from_feature_selection(theme, percentage)
		combined_dataset.append(data)
	return combined_dataset, targets	

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	combined_dataset, targets = combine_data_from_feature_selection(0.9)

	std = StandardizedData(targets)
	dataset = std.standardize_dataset(combined_dataset)  

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	cross_validation(dataset, targets, fusion_algorithm, ids)

	optimize = raw_input('Optimise parameters? y or n')
	if optimize == 'y':
			opt_params = OptimizeParameters(dataset, targets)
			opt_params.category_optimize_parameters()
