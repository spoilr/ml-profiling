print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from split_dataset import *
from feature_entropy import *
from join_attributes import *
from selected_features import *
from project_data import *
from svms import svm_subset_features

import math as math
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score

# each fold has selected features
# after cv, choose the features that were selected at least N% of the repeated cross validation process (50%, 70%, 90%)
def features_cross_validation(known_dataset, known_targets, features):
	kf = StratifiedKFold(known_targets, n_folds=10)
	error_rates = 0
	cv_features = []
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		
		error_rate, selected_features = selected_feature_one_fold(X_train, y_train, X_test, y_test, features)
		error_rates += error_rate
		
		cv_features.append(selected_features)

	print 'Final error rate in features_cross_validation %f' % (float(error_rates) / kf.n_folds)

	# for testing
	print 'CV_FEATURES %s' % str(cv_features)
	for x in cv_features:
		print len(x)	

	return cv_features	

# to be considered, a feature must appear at least 'percentage' of times in cv
def select_final_features_from_cv(cv_features, percentage):
	final_features = set()
	unique_features = set.union(*cv_features)
	nr_times = math.floor(percentage * len(cv_features))
	
	for feat in unique_features:
		if len([x for x in cv_features if feat in x]) >= nr_times:
			final_features.add(feat)

	return final_features	


def selected_feature_one_fold(X_train, y_train, X_test, y_test, features):
	# train to get the selected features
	selected_features = feature_context(X_train, y_train, features)
	train_sf = SelectedFeatures(X_train, y_train, selected_features, features)
	test_sf = SelectedFeatures(X_test, y_test, selected_features, features)
	train_dataset_of_selected_features = train_sf.extract_data_from_selected_features()
	test_dataset_of_selected_features = test_sf.extract_data_from_selected_features()
	
	# check that there are the same number of examples (only features are removed)
	assert X_test.shape[0] ==  test_dataset_of_selected_features.shape[0]

	# test the selected features
	error_rate = one_fold_measures(train_dataset_of_selected_features, test_dataset_of_selected_features, y_train, y_test)

	return error_rate, selected_features

def one_fold_measures(X_train, X_test, y_train, y_test):
	model = svm_subset_features(X_train, y_train)
	print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	# error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	error_rate = f1_score(y_test, y_pred)
	return error_rate		

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme(sys.argv[1])
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	
	known_targets = np.asarray(known_targets)

	cv_features = features_cross_validation(known_dataset, known_targets, features)

	print select_final_features_from_cv(cv_features, 0.9)
	print '######################'
	print select_final_features_from_cv(cv_features, 0.7)
	print '######################'
	print select_final_features_from_cv(cv_features, 0.5)