"""
SVM C-Support Vector Classification
Single SVM
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from split_dataset import *

from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score

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
	return error_rate
		
def measures(y_test, y_pred):
	print confusion_matrix(y_test, y_pred)
	#print(classification_report(y_test, y_pred, target_names=['highvalue','civilian']))
	print 'f1 %s' % np.array_str(f1_score(y_test, y_pred, average=None))
	print 'precision %s' % np.array_str(precision_score(y_test, y_pred, average=None))
	print 'recall %s' % np.array_str(recall_score(y_test, y_pred, average=None))

	print y_test
	print y_pred

def svm(dataset, targets):
	model = SVC()
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

if __name__ == "__main__":
	spreadsheet = Spreadsheet('/home/user/Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme(sys.argv[1])
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	known_targets = np.asarray(known_targets)

	cross_validation(known_dataset, known_targets)