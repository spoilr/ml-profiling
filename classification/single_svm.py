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
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from svms import svm_all_vars

from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

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

	print '########## MISCLASSIFIED ########## %d %s' % (len(misclassified_ids), str(misclassified_ids))
	
	print 'Final error rate %f' % (float(error_rates) / kf.n_folds)	

def one_fold_measures(X_train, X_test, y_train, y_test):
	model = svm_all_vars(X_train, y_train)
	print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	# error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	error_rate = f1_score(y_test, y_pred)
	measures(y_test, y_pred)

	return error_rate, model	

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		std = StandardizedData(targets, dataset)
		known_dataset_scaled, known_targets = std.split_and_standardize_dataset()

		cross_validation(known_dataset_scaled, known_targets, ids)
		
	except IndexError:
		print "Error!! Pass 'all' as argument"

	optimize = raw_input('Optimise parameters? y or n')
	if optimize == 'y':
		score = raw_input('Score? f1, accuracy, recall, precision')
		opt_params = OptimizeParameters(known_dataset_scaled, known_targets)
		opt_params.all_optimize_parameters(score)




			