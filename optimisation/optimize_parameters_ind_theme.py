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
from standardized_data import *
from project_data import *
from binary_classification_measures import *
from best_svm import *
from selected_features import *
from thematic_data_combined import select_features

import math as math
import numpy as np
import itertools
from sklearn.cross_validation import StratifiedKFold


def cross_validation(known_dataset, known_targets, ids, current_svm):
	kf = StratifiedKFold(known_targets, n_folds=10)
	f1_scores = 0
	error_rates = 0
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = one_fold_measures(X_train, X_test, y_train, y_test, current_svm)
		f1_scores += f1
		error_rates += error_rate

	print 'Final f1 %f' % (float(f1_scores) / kf.n_folds)
	print 'Final error %f' % (float(error_rates) / kf.n_folds)
	print '################'
	return (float(f1_scores) / kf.n_folds), (float(error_rates) / kf.n_folds)
	

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

	# cv_features = features_cross_validation(known_dataset, known_targets, features, current_svm)
	# selected_features = select_final_features_from_cv(cv_features, percentage)
	selected_features = select_features(percentage, theme)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)
	combined_dataset = sf.extract_data_from_selected_features()

	std = StandardizedData(known_targets, combined_dataset)
	known_dataset_scaled, known_targets = std.split_and_standardize_dataset()  

	print '####### FEATURES ####### %d \n %s' % (len(selected_features), str(selected_features)) 	
	return cross_validation(np.array(known_dataset_scaled), known_targets, ids, current_svm)


def params():
	begin = 0.1
	end = 3
	C_range = np.arange(begin, end, 0.1)
	gamma_range = np.arange(begin, 1.3, 0.1)
	return C_range, gamma_range



if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids
	theme = raw_input("Theme.\n")
	percentage = float(raw_input("Percentage. 0.9, 0.7 or 0.5\n"))

	C_range, gamma_range = params()	
	
	for pair in itertools.product(C_range, gamma_range):
		c = pair[0]
		g = pair[1]
		current_svm = BestFeatureSVM(c, g)

		f1, error= cv(theme, percentage, current_svm)
		
		if error <= 0.33 and f1 > 0:
			with open("result.txt", "a") as myfile:	
				myfile.write('\n##############################\n')
			with open("result.txt", "a") as myfile:
				myfile.write(current_svm.to_string())
			with open("result.txt", "a") as myfile:	
				myfile.write('\nerror_maj %f' % error)
    	
		print current_svm.to_string()

	
