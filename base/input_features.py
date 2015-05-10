"""
Use features from SPSS
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
from project_data import project_data_file
from load_data import *
from parse_theme import *
from standardized_data import *
from split_dataset import *
from parameters import TOP_FEATURES_PERCENTAGE_THRESHOLD
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
from fusion import cv10
from fusion import lr
from fusion import dt
from fusion import knn
from svms import svm_selected_for_features_fusion
import math as math

themes = ['net', 'ill', 'ideo']

def select_proxy_features(theme, target, nr_times):
	if theme == 'ill':
		if target == 'civil':
			return get_best(civil_ill, civil_ill_x, civil_ill_y, nr_times)
		else:
			return get_best(highval_ill, highval_ill_x, highval_ill_y, nr_times)
	if theme == 'ideo':
		if target == 'civil':
			return get_best(civil_ideo, civil_ideo_x, civil_ideo_y, nr_times)
		else:
			return get_best(highval_ideo, highval_ideo_x, highval_ideo_y, nr_times)
	if theme == 'net':
		if target == 'civil':
			return get_best(civil_net, civil_net_x, civil_net_y, nr_times)
		else:
			return get_best(highval_net, highval_net_x, highval_net_y, nr_times)

def thematic_data_from_feature_selection(orig_targets, theme, target):
	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, orig_targets)
	
	nr_times = int(math.floor(TOP_FEATURES_PERCENTAGE_THRESHOLD * len(features)))

	known_targets = np.asarray(known_targets)
	ssa_features = select_proxy_features(theme, target, nr_times)
	sf = SelectedFeatures(known_dataset, known_targets, ssa_features, features)

	print '####### %s FEATURES ####### %d %s' % (theme, len(ssa_features), str(ssa_features)) 

	return sf.extract_data_from_selected_features(), known_targets

def combine_data_from_feature_selection(orig_targets, target):
	combined_dataset = []
	targets = []
	for theme in themes:
		data, targets = thematic_data_from_feature_selection(orig_targets, theme, target)
		combined_dataset.append(data)
	return combined_dataset, targets	

def combine_and_cv(targets, target, tech):
	combined_dataset, targets = combine_data_from_feature_selection(targets, target)

	if tech == 'lr':
		cv10(combined_dataset, targets, fusion_algorithm, ids, lr)
	elif tech == 'dt':
		cv10(combined_dataset, targets, fusion_algorithm, ids, dt)
	elif tech == 'knn':
		std = StandardizedData(targets)
		combined_dataset = std.standardize_dataset(combined_dataset)  
		cv10(combined_dataset, targets, fusion_algorithm, ids, knn)
	elif tech == 'svm':
		std = StandardizedData(targets)
		combined_dataset = std.standardize_dataset(combined_dataset)  
		cv10(combined_dataset, targets, fusion_algorithm, ids, svm_selected_for_features_fusion, ind=True)
	else:
		print 'ERROR technique'	

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")

	tech = raw_input("Enter algorithm. Choose between lr, dt, knn, svm")

	print '########## HIGHVAL ##########'
	combine_and_cv(targets, 'highval', tech)

	print '########## CIVIL ##########'
	combine_and_cv(targets, 'civil', tech)

	

	
