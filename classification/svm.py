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
from binary_classification_measures import *

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
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

def cross_validation(known_dataset, known_targets, fusion_algorithm):
	kf = StratifiedKFold(known_targets, n_folds=5)
	error_rates = 0
	# cross validation
	for train_index, test_index in kf:
		error_rate = fusion_outputs(known_dataset, known_targets, train_index, test_index, fusion_algorithm)
		
		error_rates += error_rate
	print 'Final error rate %f' % (float(error_rates) / kf.n_folds)
		
def fusion_outputs(known_dataset, known_targets, train_index, test_index, fusion_algorithm):
	
	combined_predictions = []
	y_test = []

	if fusion_algorithm == 'maj':
		predictions, y_test, accuracies = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index)
		combined_predictions = majority_vote(predictions, y_test, accuracies)

	elif fusion_algorithm == 'wmaj':
		predictions, y_test, accuracies = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index)
		combined_predictions = weighted_majority(predictions, y_test)

	elif fusion_algorithm == 'svm':
		y_test, predictions, combined_predictions = svm_fusion(known_dataset, known_targets, train_index, test_index)

	elif fusion_algorithm == 'nn':
		print 'not done'
	else:
		print 'Error parsing algorithm'

	print '###############'
	print predictions
	print y_test
	print combined_predictions
	print '###############'

	measures(y_test, combined_predictions)
	return (float(sum((combined_predictions - y_test)**2)) / len(y_test))	

# Training and testing sets initially
# 2/3 are used to train the SVM and 1/3 is used to train(after the output is obtained) the fusion SVM
def svm_fusion(known_dataset, known_targets, train_index, test_index):
	training_predictions = []
	predictions = []
	fusion_Y_train = []
	y_train, final_y_test = known_targets[train_index], known_targets[test_index]

	kf = StratifiedKFold(y_train, n_folds=3)
	curr = 0
	for inner_train_index, inner_test_index in kf:

		for i in range(0, NR_THEMES):
			X_train, final_X_test = known_dataset[i][train_index], known_dataset[i][test_index]
			svm_X_train, svm_Y_train = X_train[inner_train_index], y_train[inner_train_index]
			fusion_X_train, fusion_Y_train = X_train[inner_test_index], y_train[inner_test_index]

			model = svm(svm_X_train, svm_Y_train)
			training_predictions.append(model.predict(fusion_X_train))
			predictions.append(model.predict(final_X_test))

		curr+=1
		if curr == 1:
			break

	training_pred_input = np.vstack(training_predictions).T
	fusion_model = svm(training_pred_input, fusion_Y_train)

	pred_input = np.vstack(predictions).T
	combined_predictions = fusion_model.predict(pred_input)

	return final_y_test, predictions, combined_predictions.tolist()

def combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index):
	predictions = []
	accuracies = []
	y_train, y_test = known_targets[train_index], known_targets[test_index]
	for i in range(0, NR_THEMES):
		X_train, X_test = known_dataset[i][train_index], known_dataset[i][test_index]
		model = svm(X_train, y_train)
		accuracy = model.score(X_test, y_test)
		print 'Model score for %s is %f' % (themes[i], accuracy)
		y_pred = model.predict(X_test)
		predictions.append(y_pred)
		accuracies.append(accuracy)
	
	predictions = np.array((predictions[0], predictions[1], predictions[2]), dtype=float)
	return predictions, y_test, accuracies

def svm(dataset, targets):
	model = SVC(class_weight='auto', C=0.7)
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

	net_scaled = preprocessing.scale(net[0])
	ill_scaled = preprocessing.scale(ill[0])
	ideo_scaled = preprocessing.scale(ideo[0])

	dataset.append(net_scaled)
	dataset.append(ill_scaled)
	dataset.append(ideo_scaled)

	# known targets should be all the same for all themes
	assert np.array_equal(net[1], ill[1])
	assert np.array_equal(net[1], ideo[1])
	return dataset, net[1]


if __name__ == "__main__":
	spreadsheet = Spreadsheet('/home/user/Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	targets = data.targets

	dataset, targets = get_thematic_data()
	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	cross_validation(dataset, targets, fusion_algorithm)

	optimize = raw_input('Optimise parameters? y or n')
	if optimize == 'y':
		C_range = 0.5 * np.array(range(1, 1000))
		gamma_range = 0.5 * np.array(range(1, 1000))
		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = StratifiedKFold(y=targets, n_folds=3)
		grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=cv)
		
		for i in range(0, NR_THEMES):
			categ_dataset = dataset[i]	
			grid.fit(categ_dataset, targets)
			print("Best classifier: ", grid.best_estimator_)	