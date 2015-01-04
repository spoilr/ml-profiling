"""
SVM C-Support Vector Classification
Combine SVM for themes
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from split_dataset import *
from labels_fusion import *

from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from collections import Counter

NR_THEMES = 3
themes = ['net', 'ill', 'ideo']

def cross_validation(known_dataset, known_targets):
	kf = StratifiedKFold(known_targets, n_folds=10)
	error_rates = 0
	# cross validation
	for train_index, test_index in kf:
		error_rate = combine_predictions_one_fold(known_dataset, known_targets, train_index, test_index)
		
		error_rates += error_rate
	print 'Final error rate %f' % (float(error_rates) / kf.n_folds)
		
def combine_predictions_one_fold(known_dataset, known_targets, train_index, test_index):
	predictions = []
	accuracies = []
	y_train, y_test = known_targets[train_index], known_targets[test_index]
	for i in range(0,NR_THEMES):
		X_train, X_test = known_dataset[i][train_index], known_dataset[i][test_index]
		model = svm(X_train, y_train)
		accuracy = model.score(X_test, y_test)
		print 'Model score for %s is %f' % (themes[i], accuracy)
		y_pred = model.predict(X_test)
		predictions.append(y_pred)
		accuracies.append(accuracy)
	
	predictions = np.array((predictions[0], predictions[1], predictions[2]), dtype=float)

	combined_predictions = majority_vote(predictions, y_test, accuracies)

	print predictions
	print y_test
	print combined_predictions
	return (float(sum((combined_predictions - y_test)**2)) / len(y_test))

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

def get_known_data_from_theme(theme):
	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	known_targets = np.asarray(known_targets)
	return [known_dataset, known_targets]

def get_thematic_data():
	dataset = []

	net = get_known_data_from_theme(themes[0])
	ill = get_known_data_from_theme(themes[1])
	ideo = get_known_data_from_theme(themes[2])
	dataset.append(net[0])
	dataset.append(ill[0])
	dataset.append(ideo[0])

	# known targets should be all the same for all themes
	assert np.array_equal(net[1], ill[1])
	assert np.array_equal(net[1], ideo[1])
	return dataset, net[1]

if __name__ == "__main__":
	spreadsheet = Spreadsheet('/home/user/Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	targets = data.targets

	dataset, targets = get_thematic_data()
	cross_validation(dataset, targets)


