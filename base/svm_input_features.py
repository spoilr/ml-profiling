"""
SVM C-Support Vector Classification
Combine SVM for themes
Feature selection is applied before
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
from project_data import project_data_file
from load_data import *
from parse_theme import *
from split_dataset import *
sys.path.insert(0, 'classification/')
from parameters import TOP_FEATURES_PERCENTAGE_THRESHOLD
from svm_fusion import *
from standardized_data import *
from selected_features import *
from closest_distance import *
from ssa_features import civil_ideo
from ssa_features import civil_ideo_x
from ssa_features import civil_ideo_y
from ssa_features import civil_ill
from ssa_features import civil_ill_x
from ssa_features import civil_ill_y
from ssa_features import civil_net
from ssa_features import civil_net_x
from ssa_features import civil_net_y
from ssa_features import highval_ideo
from ssa_features import highval_ideo_x
from ssa_features import highval_ideo_y
from ssa_features import highval_ill
from ssa_features import highval_ill_x
from ssa_features import highval_ill_y
from ssa_features import highval_net
from ssa_features import highval_net_x
from ssa_features import highval_net_y
import math as math

themes = ['net', 'ill', 'ideo']

def select_proxy_features(theme, target, nr_times):
	if theme == 'ill':
		if target == 'civil':
			return get_best(civil_all, civil_all_x, civil_all_y, nr_times)
		else:
			return get_best(highval_all, highval_all_x, highval_all_y, nr_times)
	if theme == 'ideo':
		if target == 'civil':
			return get_best(civil_all, civil_all_x, civil_all_y, nr_times)
		else:
			return get_best(highval_all, highval_all_x, highval_all_y, nr_times)
	if theme == 'net':
		if target == 'civil':
			return get_best(civil_all, civil_all_x, civil_all_y, nr_times)
		else:
			return get_best(highval_all, highval_all_x, highval_all_y, nr_times)

def thematic_data_from_feature_selection(orig_targets, theme, percentage, target):
	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, orig_targets)
	
	nr_times = int(math.floor(TOP_FEATURES_PERCENTAGE_THRESHOLD * len(features)))

	known_targets = np.asarray(known_targets)
	ssa_features = select_proxy_features(theme, target, nr_times)
	sf = SelectedFeatures(known_dataset, known_targets, ssa_features, features)

	print '####### %s FEATURES ####### %d %s' % (theme, len(ssa_features), str(ssa_features)) 

	return sf.extract_data_from_selected_features(), known_targets

def combine_data_from_feature_selection(orig_targets, percentage, target):
	combined_dataset = []
	targets = []
	for theme in themes:
		data, targets = thematic_data_from_feature_selection(orig_targets, theme, percentage, target)
		combined_dataset.append(data)
	return combined_dataset, targets	

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	print '########## CIVIL #####'
	combined_dataset, targets = combine_data_from_feature_selection(targets, 1, 'civil')

	print '########## HIGHVAL #####'
	combined_dataset, targets = combine_data_from_feature_selection(targets, 1, 'highval')

	std = StandardizedData(targets)
	dataset = std.standardize_dataset(combined_dataset)  

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	cross_validation(dataset, targets, fusion_algorithm, ids)
