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

	print precision(y_test, y_pred)
	print recall(y_test, y_pred)
	print f1(y_test, y_pred)
	print accuracy(y_test, y_pred)

	print y_test
	print y_pred

if __name__ == "__main__":
	# y_pred = [2, 1, 2, 2]
	# y_true = [2, 1, 2, 1]
	y_true = [1, 1, 2, 2, 2, 2, 2, 2]
	y_pred = [2, 2, 2, 2, 2, 2, 2, 2]
	print precision(y_true, y_pred)
	print recall(y_true, y_pred)
	print f1(y_true, y_pred)
	print accuracy(y_true, y_pred)
	print accuracy_number(y_true, y_pred)