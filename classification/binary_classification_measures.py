"""
Binary classification measures. 
Includes precision, recall, f1, accuracy, and accuracy number.
"""

print(__doc__)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

### None - scores for each class are returned

def precision(y_true, y_pred):
	return precision_score(y_true, y_pred, average=None)

def recall(y_true, y_pred):
	return recall_score(y_true, y_pred, average=None)

def f1(y_true, y_pred):
	return f1_score(y_true, y_pred, average=None)

def accuracy(y_true, y_pred):
	return accuracy_score(y_true, y_pred)

def accuracy_number(y_true, y_pred):
	return accuracy_score(y_true, y_pred, normalize=False)

def measures(y_test, y_pred):
	print confusion_matrix(y_test, y_pred)
	#print(classification_report(y_test, y_pred, target_names=['highvalue','civilian']))

	print 'PRECISION %s' % str(precision(y_test, y_pred))
	print 'RECALL %s' % str(recall(y_test, y_pred))
	print 'F1 %s' % str(f1(y_test, y_pred))
	print 'ACCURACY %s' % str(accuracy(y_test, y_pred))

	print 'Y_TEST %s' % str(y_test)
	print 'Y_PRED %s' % str(y_pred)

if __name__ == "__main__":
	# y_pred = [2, 1, 2, 2]
	# y_true = [2, 1, 2, 1]
	y_true = [1, 1, 2, 2, 2, 2, 2, 2]
	y_pred = [2, 2, 2, 2, 2, 2, 2, 2]
	print 'PRECISION %s' % str(precision(y_true, y_pred))
	print 'RECALL %s' % str(recall(y_true, y_pred))
	print 'F1 %s' % str(f1(y_true, y_pred))
	print 'ACCURACY %s' % str(accuracy(y_true, y_pred))
	print 'ACCURACY %f' % accuracy_number(y_true, y_pred)