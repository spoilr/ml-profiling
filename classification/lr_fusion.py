import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from split_dataset import *
from labels_fusion import *
from binary_classification_measures import *
from standardized_data import *
from misclassified_ids import *
from project_data import *

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter

NR_THEMES = 3
themes = ['net', 'ill', 'ideo']

def cross_validation(known_dataset, known_targets, fusion_algorithm, ids):
	misclassified_ids = []

	kf = StratifiedKFold(known_targets, n_folds=10)
	f1_scores = 0
	error_rates = 0
	hp_rates = 0
	hr_rates = 0
	hf_rates = 0
	cp_rates = 0
	cr_rates = 0
	cf_rates = 0
	# cross validation
	for train_index, test_index in kf:
		error, f1, mis_ids, (hp, hr, hf), (cp, cr, cf) = fusion_outputs(known_dataset, known_targets, train_index, test_index, fusion_algorithm, ids)
		
		f1_scores += f1
		error_rates += error
		
		hp_rates += hp
		hr_rates += hr
		hf_rates += hf
		cp_rates += cp
		cr_rates += cr
		cf_rates += cf
		misclassified_ids += mis_ids


	misclassified_ids = set(misclassified_ids)	
	# print '########## MISCLASSIFIED ########## %d \n %s' % (len(misclassified_ids), str(misclassified_ids))
	print 'Final f1 %f' % (float(f1_scores) / kf.n_folds)
	print 'Final error %f' % (float(error_rates) / kf.n_folds)

	print 'Highval precision %f' % (float(hp_rates) / kf.n_folds)
	print 'Highval recall %f' % (float(hr_rates) / kf.n_folds)
	print 'Highval f1 %f' % (float(hf_rates) / kf.n_folds)
	print 'Civil precision %f' % (float(cp_rates) / kf.n_folds)
	print 'Civil recall %f' % (float(cr_rates) / kf.n_folds)
	print 'Civil f1 %f' % (float(cf_rates) / kf.n_folds)

		
def fusion_outputs(known_dataset, known_targets, train_index, test_index, fusion_algorithm, ids):
	misclassified_ids = []
	combined_predictions = []
	y_test = []

	if fusion_algorithm == 'maj':
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids)
		combined_predictions = majority_vote(predictions, y_test, accuracies)

	elif fusion_algorithm == 'wmaj':
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids)
		combined_predictions = weighted_majority(predictions, y_test)

	elif fusion_algorithm == 'svm':
		y_test, predictions, combined_predictions, misclassified_ids = svm_fusion(known_dataset, known_targets, train_index, test_index, ids)

	elif fusion_algorithm == 'nn':
		print 'not done'
	else:
		print 'Error parsing algorithm'

	print '###############'
	print 'Y_PRED %s' % str(predictions)
	print 'Y_TEST %s' % str(y_test)
	print 'COMBINED %s' % str(combined_predictions)
	print '###############'

	(hp, hr, hf), (cp, cr, cf) = measures(y_test, combined_predictions)

	error = (float(sum((combined_predictions - y_test)**2)) / len(y_test))
	f1 = f1_score(combined_predictions, y_test)
	return error, f1, misclassified_ids, (hp, hr, hf), (cp, cr, cf)

# Training and testing sets initially
# 2/3 are used to train the SVM and 1/3 is used to train(after the output is obtained) the fusion SVM
def svm_fusion(known_dataset, known_targets, train_index, test_index, ids):
	misclassified_ids = []

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

			model = lr(svm_X_train, svm_Y_train)
			training_predictions.append(model.predict(fusion_X_train))
			predictions.append(model.predict(final_X_test))
			misclassified_ids += add_misclassified_ids(model, test_index, known_dataset[i], known_targets, ids)

		curr+=1
		if curr == 1:
			break

	training_pred_input = np.vstack(training_predictions).T
	fusion_model = inner_svm(training_pred_input, fusion_Y_train)

	pred_input = np.vstack(predictions).T
	combined_predictions = fusion_model.predict(pred_input)

	return final_y_test, predictions, combined_predictions.tolist(), misclassified_ids

def inner_svm(dataset, targets):
	model = SVC(class_weight='auto')
	model.fit(dataset, targets)
	return model

def lr(dataset, targets):
	model = LogisticRegression(class_weight='auto')
	model.fit(dataset, targets)
	return model

# called majority because it is used in both cases of majority_voting and weighted_majority voting.
def combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids):
	misclassified_ids = []

	predictions = []
	accuracies = []
	y_train, y_test = known_targets[train_index], known_targets[test_index]
	for i in range(0, NR_THEMES):
		X_train, X_test = known_dataset[i][train_index], known_dataset[i][test_index]
		model = lr(X_train, y_train)
		accuracy = model.score(X_test, y_test)
		print 'Model score for %s is %f' % (themes[i], accuracy)
		y_pred = model.predict(X_test)
		predictions.append(y_pred)
		accuracies.append(accuracy)
		misclassified_ids += add_misclassified_ids(model, test_index, known_dataset[i], known_targets, ids)
	
	predictions = np.array((predictions[0], predictions[1], predictions[2]), dtype=float)
	return predictions, y_test, accuracies, misclassified_ids
