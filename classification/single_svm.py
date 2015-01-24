"""
SVM C-Support Vector Classification
Single SVM
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from binary_classification_measures import *
from optimize_parameters import *
from standardized_data import *
from misclassified_ids import *
sys.path.insert(0, 'feature context/')
from feature_selection_cv import *

from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report

def cross_validation(known_dataset, known_targets, ids):
	misclassified_ids = []

	kf = StratifiedKFold(known_targets, n_folds=10)
	error_rates = 0
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		error_rate, model = one_fold_measures(X_train, X_test, y_train, y_test)
		error_rates += error_rate
		misclassified_ids += add_misclassified_ids(model, test_index, known_dataset, known_targets, ids)

	print '########## MISCLASSIFIED ########## %d' % len(misclassified_ids)
	print misclassified_ids	
	print 'Final error rate %f' % (float(error_rates) / kf.n_folds)	

def one_fold_measures(X_train, X_test, y_train, y_test):
	model = svm(X_train, y_train)
	print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))

	measures(y_test, y_pred)

	return error_rate, model

def svm(dataset, targets):
	model = SVC()
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def feature_selection_before(val, features, targets, dataset, percentage, ids):
	if val:
		[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
		
		known_targets = np.asarray(known_targets)

		cv_features = features_cross_validation(known_dataset, known_targets, features)
		selected_features = select_final_features_from_cv(cv_features, percentage)

		sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)
		combined_dataset = sf.extract_data_from_selected_features()

		std = StandardizedData(known_targets)
		combined_dataset = std.standardize_dataset(combined_dataset)  

		cross_validation(np.array(combined_dataset), known_targets, ids)
	else:
		std = StandardizedData(targets, dataset)
		known_dataset_scaled, known_targets = std.split_and_standardize_dataset()

		cross_validation(known_dataset_scaled, known_targets, ids)	

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		feature_selection_before(True, features, targets, dataset, 0.9, ids)
		
	except IndexError:
		print "Error!! Pass 'all' as argument"

	optimize = raw_input('Optimise parameters? y or n')
	if optimize == 'y':
		opt_params = OptimizeParameters(dataset, targets)
		opt_params.all_optimize_parameters()




			