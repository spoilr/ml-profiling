"""
Optimise parameters.
Optimise SVM for each theme individually.
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
sys.path.insert(0, 'feature context/')
from load_data import *
from parse_theme import *
from split_dataset import *
from feature_entropy import *
from join_attributes import *
from selected_features import *
from standardized_data import *
from project_data import *
from svms import svm_subset_features
from binary_classification_measures import *
from best_svm import *

import math as math
import numpy as np
from sklearn.cross_validation import StratifiedKFold

# each fold has selected features
# after cv, choose the features that were selected at least N% of the repeated cross validation process (50%, 70%, 90%)
def features_cross_validation(known_dataset, known_targets, features, current_svm):
	std = StandardizedData(known_targets, known_dataset)
	known_dataset, known_targets = std.split_and_standardize_dataset()
	kf = StratifiedKFold(known_targets, n_folds=10)
	error_rates = 0
	cv_features = []
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		
		error_rate, selected_features = selected_feature_one_fold(X_train, y_train, X_test, y_test, features, current_svm)
		error_rates += error_rate
		
		cv_features.append(selected_features)
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


def selected_feature_one_fold(X_train, y_train, X_test, y_test, features, current_svm):
	# train to get the selected features
	selected_features = feature_context(X_train, y_train, features)
	train_sf = SelectedFeatures(X_train, y_train, selected_features, features)
	test_sf = SelectedFeatures(X_test, y_test, selected_features, features)
	train_dataset_of_selected_features = train_sf.extract_data_from_selected_features()
	test_dataset_of_selected_features = test_sf.extract_data_from_selected_features()
	
	# check that there are the same number of examples (only features are removed)
	assert X_test.shape[0] ==  test_dataset_of_selected_features.shape[0]

	# test the selected features
	error_rate = features_one_fold_measures(train_dataset_of_selected_features, test_dataset_of_selected_features, y_train, y_test, current_svm)

	return error_rate, selected_features

def features_one_fold_measures(X_train, X_test, y_train, y_test, current_svm):
	model = current_svm.svm_subset_features(X_train, y_train)
	print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	return error_rate		

def cross_validation(known_dataset, known_targets, ids, current_svm):
	kf = StratifiedKFold(known_targets, n_folds=10)
	f1_scores = 0
	error_rates = 0
	hf_rates = 0
	cf_rates = 0
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = one_fold_measures(X_train, X_test, y_train, y_test, current_svm)
		f1_scores += f1
		error_rates += error_rate

		hf_rates += hf
		cf_rates += cf

	print 'Final f1 %f' % (float(f1_scores) / kf.n_folds)
	print 'Final error %f' % (float(error_rates) / kf.n_folds)
	print '################'
	return (float(f1_scores) / kf.n_folds), (float(error_rates) / kf.n_folds), (float(hf_rates) / kf.n_folds), (float(cf_rates) / kf.n_folds)
	

def one_fold_measures(X_train, X_test, y_train, y_test, current_svm):
	model = current_svm.svm_for_features_fusion(X_train, y_train)
	print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	f1 = f1_score(y_test, y_pred)
	(hp, hr, hf), (cp, cr, cf) = measures(y_test, y_pred)

	return error_rate, f1, model, (hp, hr, hf), (cp, cr, cf)		

def cv(theme, percentage, current_svm):
	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	known_targets = np.asarray(known_targets)
	cv_features = features_cross_validation(known_dataset, known_targets, features, current_svm)
	selected_features = select_final_features_from_cv(cv_features, percentage)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)
	combined_dataset = sf.extract_data_from_selected_features()

	std = StandardizedData(known_targets, combined_dataset)
	known_dataset_scaled, known_targets = std.split_and_standardize_dataset()  

	print '####### FEATURES ####### %d \n %s' % (len(selected_features), str(selected_features)) 	
	return cross_validation(np.array(known_dataset_scaled), known_targets, ids, current_svm)


def params():
	begin = 0.1
	end = 2.7
	C_range = np.arange(begin, end, 0.4)
	gamma_range = np.arange(begin, 1.3, 0.4)
	return C_range, gamma_range



if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids
	theme = raw_input("Theme.\n")
	percentage = float(raw_input("Percentage. 0.9, 0.7 or 0.5\n"))

	C_range, gamma_range = params()
	for pair in itertools.product(itertools.product(C_range,C_range), itertools.product(gamma_range,gamma_range)):
		c_subset = pair[0][0]
		g_subset = pair[1][0]
		c_fusion = pair[0][1]
		g_fusion = pair[1][1]
		current_svm = BestSVM(c_subset, g_subset, c_fusion, g_fusion)

		f1, error, hf, cf = cv(theme, percentage, current_svm)
		
		if hf >= 0.4 and cf >= 0.4 and error <= 0.3:
			with open("result.txt", "a") as myfile:	
				myfile.write('\n##############################\n')
			with open("result.txt", "a") as myfile:
				myfile.write(current_svm.to_string())
			with open("result.txt", "a") as myfile:	
				myfile.write('\nerror_maj %f' % error)
    	
		print current_svm.to_string()

	



