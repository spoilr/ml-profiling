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
from sklearn.metrics import classification_report

### None - scores for each class are returned

def tp(y_true, y_pred):
	highval_tp = len([1 for (a,b) in zip(y_true, y_pred) if a == b and a == 1])
	civil_tp = len([1 for (a,b) in zip(y_true, y_pred) if a == b and a == 2])
	return highval_tp, civil_tp

def fn(y_true, y_pred):
	highval_fn = len([1 for (a,b) in zip(y_true, y_pred) if a == 1 and b == 2])
	civil_fn = len([1 for (a,b) in zip(y_true, y_pred) if a == 2 and b == 1])
	return highval_fn, civil_fn

def fp(y_true, y_pred):
	highval_fp = len([1 for (a,b) in zip(y_true, y_pred) if a == 2 and b == 1])
	civil_fp = len([1 for (a,b) in zip(y_true, y_pred) if a == 1 and b == 2])
	return highval_fp, civil_fp

def precision(y_true, y_pred):
	# return precision_score(y_true, y_pred, average=None)
	highval_tp, civil_tp = tp(y_true, y_pred)
	highval_fp, civil_fp = fp(y_true, y_pred)

	if highval_tp + highval_fp > 0:
		highval_p = float(highval_tp) / (highval_tp + highval_fp)
	else:
		highval_p = 0

	if civil_tp + civil_fp > 0:		
		civil_p = float(civil_tp) / (civil_tp + civil_fp)
	else:
		civil_p = 0	

	return highval_p, civil_p

def recall(y_true, y_pred):
	# return recall_score(y_true, y_pred, average=None)
	highval_tp, civil_tp = tp(y_true, y_pred)
	highval_fn, civil_fn = fn(y_true, y_pred)

	if highval_tp + highval_fn > 0:
		highval_r = float(highval_tp) / (highval_tp + highval_fn)
	else:
		highval_r = 0

	if civil_tp + civil_fn > 0:		
		civil_r = float(civil_tp) / (civil_tp + civil_fn)
	else:
		civil_r = 0	

	return highval_r, civil_r

def f1(y_true, y_pred):
	# return f1_score(y_true, y_pred, average=None)
	highval_p, civil_p = precision(y_true, y_pred)
	highval_r, civil_r = recall(y_true, y_pred)

	if highval_p+highval_r > 0:
		highval_f1 = 2 * float(highval_p*highval_r) / (highval_p+highval_r)
	else:
		highval_f1 = 0

	if civil_p+civil_r > 0:		
		civil_f1 = 2 * float(civil_p*civil_r) / (civil_p+civil_r)
	else:
		civil_f1 = 0	

	return highval_f1, civil_f1 

def accuracy(y_true, y_pred):
	return accuracy_score(y_true, y_pred)

def accuracy_number(y_true, y_pred):
	return accuracy_score(y_true, y_pred, normalize=False)

def measures(y_test, y_pred):
	print confusion_matrix(y_test, y_pred)
	print(classification_report(y_test, y_pred, target_names=['highvalue','civilian']))

	p = precision(y_test, y_pred)
	r = recall(y_test, y_pred)
	f = f1(y_test, y_pred)
	a = accuracy(y_test, y_pred)

	print 'PRECISION %s' % str(p)
	print 'RECALL %s' % str(r)
	print 'F1 %s' % str(f)
	print 'ACCURACY %s' % str(a)

	print 'Y_TEST %s' % str(y_test)
	print 'Y_PRED %s' % str(y_pred)

	hp, cp = p
	hr, cr = r
	hf, cf = f

	return (hp, hr, hf), (cp, cr, cf)

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