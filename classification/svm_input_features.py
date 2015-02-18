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
from ssa_features import get_decoded_features
from ssa_features import proxy_ideo
from ssa_features import proxy_net
from ssa_features import proxy_ill
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD

themes = ['net', 'ill', 'ideo']

def select_proxy_features(theme):
	if theme == 'ill':
		return get_decoded_features(proxy_ill)
	if theme == 'ideo':
		return get_decoded_features(proxy_ideo)
	if theme == 'net':
		return get_decoded_features(proxy_net)

def thematic_data_from_feature_selection(orig_targets, theme, percentage):
	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, orig_targets)
	
	known_targets = np.asarray(known_targets)
	ssa_features = select_proxy_features(theme)
	sf = SelectedFeatures(known_dataset, known_targets, ssa_features, features)

	print '####### %s FEATURES ####### %d %s' % (theme, len(ssa_features), str(ssa_features)) 

	return sf.extract_data_from_selected_features(), known_targets

def combine_data_from_feature_selection(orig_targets, percentage):
	combined_dataset = []
	targets = []
	for theme in themes:
		data, targets = thematic_data_from_feature_selection(orig_targets, theme, percentage)
		combined_dataset.append(data)
	return combined_dataset, targets	

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	combined_dataset, targets = combine_data_from_feature_selection(targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD)

	std = StandardizedData(targets)
	dataset = std.standardize_dataset(combined_dataset)  

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	cross_validation(dataset, targets, fusion_algorithm, ids)
