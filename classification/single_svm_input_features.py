"""
SVM C-Support Vector Classification
Single SVM
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from standardized_data import *
from misclassified_ids import *
from binary_classification_measures import *
from selected_features import *
from ssa_features import ssa_features
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from svms import svm_selected_vars

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score

def cross_validation(known_dataset, known_targets, ids):
	misclassified_ids = []

	kf = StratifiedKFold(known_targets, n_folds=10)
	f1_scores = 0
	error_rates = 0
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		error_rate, f1, model = one_fold_measures(X_train, X_test, y_train, y_test)
		f1_scores += f1
		error_rates += error_rate
		misclassified_ids += add_misclassified_ids(model, test_index, known_dataset, known_targets, ids)

	print '########## MISCLASSIFIED ########## %d %s' % (len(misclassified_ids), str(misclassified_ids))
	
	print 'Final f1 %f' % (float(f1_scores) / kf.n_folds)
	print 'Final error %f' % (float(error_rates) / kf.n_folds)

def one_fold_measures(X_train, X_test, y_train, y_test):
	model = svm_selected_vars(X_train, y_train)
	print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	f1 = f1_score(y_test, y_pred)
	measures(y_test, y_pred)

	return error_rate, f1, model

def feature_selection_before(features, targets, dataset, percentage, ids):
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)		
	known_targets = np.asarray(known_targets)

	sf = SelectedFeatures(known_dataset, known_targets, ssa_features, features)	
	ssa_dataset = sf.extract_data_from_selected_features()

	std = StandardizedData(known_targets)
	ssa_dataset = std.standardize_dataset(ssa_dataset)  

	cross_validation(np.array(ssa_dataset), known_targets, ids)

	print '####### FEATURES ####### %d \n %s' % (len(ssa_features), str(ssa_features))

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		feature_selection_before(features, targets, dataset, CV_PERCENTAGE_OCCURENCE_THRESHOLD, ids)
		
	except IndexError:
		print "Error!! Pass 'all' as argument"






			