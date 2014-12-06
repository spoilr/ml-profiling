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
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		one_fold(X_train, X_test, y_train, y_test)

def one_fold(X_train, X_test, y_train, y_test):
	model = svm(X_train, y_train)
	print 'Model score %d' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	print confusion_matrix(y_test, y_pred)
	#print(classification_report(y_test, y_pred, target_names=['highvalue','civilian']))
	print 'f1 %s' % np.array_str(f1_score(y_test, y_pred, average=None))
	print 'precision %s' % np.array_str(precision_score(y_test, y_pred, average=None))
	print 'recall %s' % np.array_str(recall_score(y_test, y_pred, average=None))

	print y_test
	print y_pred
	print '##############################################'
