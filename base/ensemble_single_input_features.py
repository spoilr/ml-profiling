"""
Use features from SPSS
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import project_data_file
from parse_theme import *
from standardized_data import *
from binary_classification_measures import measures
from selected_features import *
from ssa_features import civil_all
from ssa_features import civil_all_x
from ssa_features import civil_all_y
from ssa_features import highval_all
from ssa_features import highval_all_x
from ssa_features import highval_all_y
from closest_distance import get_best
from cv import cv10_ensemble
from cv import lr_one_fold_measures
from cv import dt
from cv import knn
from svms import svm_selected_vars
from parameters import TOP_FEATURES_PERCENTAGE_THRESHOLD
import math as math

def feature_selection(features, targets, dataset, target, dt, knn, svm):
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
		
	known_targets = np.asarray(known_targets)

	nr_times = int(math.floor(TOP_FEATURES_PERCENTAGE_THRESHOLD * len(features)))

	if target == 'civil':
		ssa_features = get_best(civil_all, civil_all_x, civil_all_y, nr_times)
	else:
		ssa_features = get_best(highval_all, highval_all_x, highval_all_y, nr_times)

	sf = SelectedFeatures(known_dataset, known_targets, ssa_features, features)
	ssa_dataset = sf.extract_data_from_selected_features()

	std = StandardizedData(known_targets, ssa_dataset)
	ssa_dataset_scaled, known_targets_scaled = std.split_and_standardize_dataset()  

	assert not set(known_targets).isdisjoint(known_targets_scaled)

	file_name = "ensemble_single_" + target + ".txt"
	for i in range(100):
		cv10_ensemble(ssa_dataset, known_targets, ssa_dataset_scaled, dt, knn, svm, prt=True, file_name=file_name)

	print '####### FEATURES ####### %d \n %s' % (len(ssa_features), str(ssa_features))


if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])

		print '########## CIVIL ##########'
		feature_selection(features, targets, dataset, 'civil', dt, knn, svm_selected_vars)

		print '########## HIGHVAL ##########'
		feature_selection(features, targets, dataset, 'highval', dt, knn, svm_selected_vars)

		
	except IndexError:
		print "Error!! Pass 'all' as argument"






