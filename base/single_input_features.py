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
from cv import cross_validation
from cv import lr_one_fold_measures
from cv import dt_one_fold_measures
from cv import knn_one_fold_measures
from cv import single_svm_fs_one_fold_measures
from parameters import TOP_FEATURES_PERCENTAGE_THRESHOLD
import math as math

def feature_selection(features, targets, dataset, ids, target, one_fold_measures, standardize=False):
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	assert len(known_dataset) == 92
	assert len(known_targets) == 92
		
	known_targets = np.asarray(known_targets)

	nr_times = int(math.floor(TOP_FEATURES_PERCENTAGE_THRESHOLD * len(features)))

	if target == 'civil':
		ssa_features = get_best(civil_all, civil_all_x, civil_all_y, nr_times)
	else:
		ssa_features = get_best(highval_all, highval_all_x, highval_all_y, nr_times)

	sf = SelectedFeatures(known_dataset, known_targets, ssa_features, features)
	ssa_dataset = sf.extract_data_from_selected_features()

	if standardize:
		std = StandardizedData(known_targets, ssa_dataset)
		ssa_dataset, known_targets = std.split_and_standardize_dataset()  

	cross_validation(ssa_dataset, known_targets, ids, one_fold_measures)

	print '####### FEATURES ####### %d \n %s' % (len(ssa_features), str(ssa_features))


if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])

		tech = raw_input("Enter algorithm. Choose between lr, dt, knn, svm")

		if tech == 'lr':
			print '########## CIVIL ##########'
			feature_selection(features, targets, dataset, ids, 'civil', lr_one_fold_measures)

			print '########## HIGHVAL ##########'
			feature_selection(features, targets, dataset, ids, 'highval', lr_one_fold_measures)
		elif tech == 'dt':
			print '########## CIVIL ##########'
			feature_selection(features, targets, dataset, ids, 'civil', dt_one_fold_measures)

			print '########## HIGHVAL ##########'
			feature_selection(features, targets, dataset, ids, 'highval', dt_one_fold_measures)
		elif tech == 'knn':
			print '########## CIVIL ##########'
			feature_selection(features, targets, dataset, ids, 'civil', knn_one_fold_measures)

			print '########## HIGHVAL ##########'
			feature_selection(features, targets, dataset, ids, 'highval', knn_one_fold_measures)
		elif tech == 'svm':
			print '########## CIVIL ##########'
			feature_selection(features, targets, dataset, ids, 'civil', single_svm_fs_one_fold_measures, standardize=True)

			print '########## HIGHVAL ##########'
			feature_selection(features, targets, dataset, ids, 'highval', single_svm_fs_one_fold_measures, standardize=True)
		else:
			print 'ERROR technique'	

	except IndexError:
		print "Error!! Pass 'all' as argument"






