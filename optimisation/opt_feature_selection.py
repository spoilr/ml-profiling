print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
from load_data import *
from parse_theme import *
from split_dataset import *
from feature_entropy import *
from join_attributes import *
from selected_features import *
from standardized_data import *
from project_data import *
from best_svm import *

import math as math
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score

# each fold has selected features
# after cv, choose the features that were selected at least N% of the repeated cross validation process (50%, 70%, 90%)
def features_cross_validation(known_dataset, known_targets, features, current_svm):
	std = StandardizedData(known_targets, known_dataset)
	known_dataset, known_targets = std.split_and_standardize_dataset()
	kf = StratifiedKFold(known_targets, n_folds=10)
	error_rates = 0
	f1_rates = 0
	cv_features = []
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		
		error_rate, selected_features, f1 = selected_feature_one_fold(X_train, y_train, X_test, y_test, features, current_svm)
		error_rates += error_rate
		f1_rates += f1
		
		cv_features.append(selected_features)

	return (float(error_rates) / kf.n_folds), (float(f1_rates) / kf.n_folds)	

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
	error_rate, f1 = one_fold_measures(train_dataset_of_selected_features, test_dataset_of_selected_features, y_train, y_test, current_svm)

	return error_rate, selected_features, f1

def one_fold_measures(X_train, X_test, y_train, y_test, current_svm):
	model = current_svm.svm_subset_features(X_train, y_train)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	f1 = f1_score(y_test, y_pred)
	return error_rate, f1		

def params():
	begin = 0.1
	end = 2.7
	C_range = np.arange(begin, end, 0.3)
	gamma_range = np.arange(begin, 1.3, 0.3)
	return C_range, gamma_range

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	theme = raw_input("Theme.\n")

	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	
	known_targets = np.asarray(known_targets)

	

	C_range, gamma_range = params()	
	
	for pair in itertools.product(C_range, gamma_range):
		c = pair[0]
		g = pair[1]
		current_svm = BestFeatureSVM(c, g)
		error, f1 = features_cross_validation(known_dataset, known_targets, features, current_svm)

		if error <= 0.33 and f1 > 0:
			with open("result.txt", "a") as myfile:	
				myfile.write('\n##############################\n')
			with open("result.txt", "a") as myfile:
				myfile.write(current_svm.to_string())
			with open("result.txt", "a") as myfile:	
				myfile.write('\nerror %f' % error)
			with open("result.txt", "a") as myfile:	
				myfile.write('\nf1 %f' % f1)
    	
		print current_svm.to_string() + " - " + str(error) + " - " + str(f1)

