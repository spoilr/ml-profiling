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
from cross_validation import *
from optimize_parameters import *
from standardized_data import *

from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report

def cross_validation(known_dataset, known_targets):
	kf = StratifiedKFold(known_targets, n_folds=10)
	error_rates = 0
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		error_rate = one_fold_measures(X_train, X_test, y_train, y_test)
		error_rates += error_rate
	print 'Final error rate %f' % (float(error_rates) / kf.n_folds)	

def one_fold_measures(X_train, X_test, y_train, y_test):
	model = svm(X_train, y_train)
	print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))

	measures(y_test, y_pred)

	return error_rate

def svm(dataset, targets):
	model = SVC()
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

if __name__ == "__main__":
	spreadsheet = Spreadsheet('/home/user/Downloads/ip/project data no.xlsx')
	data = Data(spreadsheet)
	targets = data.targets

	try:
		[dataset, features] = parse_theme(sys.argv[1])

		std = StandardizedData(targets, dataset)
		known_dataset_scaled, known_targets = std.split_and_standardize_dataset()

		cross_validation(known_dataset_scaled, known_targets)
	except IndexError:
		print "Error!! Pass 'all' as argument"

	optimize = raw_input('Optimise parameters? y or n')
	if optimize == 'y':
		opt_params = OptimizeParameters(dataset, targets)
		opt_params.all_optimize_parameters()