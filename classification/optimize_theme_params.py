"""
SVM C-Support Vector Classification
Combine SVM for themes
Feature selection is applied before
Optimize parameters
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
from load_data import *
from parse_theme import *
from split_dataset import *
from feature_entropy import *
from join_attributes import *
from selected_features import *
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from project_data import *
from labels_fusion import *
from standardized_data import *
from misclassified_ids import *


import math as math
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score


NR_THEMES = 3
themes = ['net', 'ill', 'ideo']

def thematic_data_from_feature_selection(orig_targets, theme, percentage, current_svm):
	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, orig_targets)
	
	known_targets = np.asarray(known_targets)

	cv_features = features_cross_validation(known_dataset, known_targets, features, current_svm)
	selected_features = select_final_features_from_cv(cv_features, percentage)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)

	return sf.extract_data_from_selected_features(), known_targets

def combine_data_from_feature_selection(orig_targets, percentage, current_svm):
	combined_dataset = []
	targets = []
	for theme in themes:
		data, targets = thematic_data_from_feature_selection(orig_targets, theme, percentage, current_svm)
		combined_dataset.append(data)
	return combined_dataset, targets	


def cross_validation(known_dataset, known_targets, fusion_algorithm, ids, current_svm):
	misclassified_ids = []

	kf = StratifiedKFold(known_targets, n_folds=10)
	f1_scores = 0
	error_rates = 0
	# cross validation
	for train_index, test_index in kf:
		error, f1, mis_ids = fusion_outputs(known_dataset, known_targets, train_index, test_index, fusion_algorithm, ids, current_svm)
		
		f1_scores += f1
		error_rates += error
		misclassified_ids += mis_ids


	misclassified_ids = set(misclassified_ids)	
	print 'Fusion algorithm %s' % fusion_algorithm
	print 'Final f1 %f' % (float(f1_scores) / kf.n_folds)
	print 'Final error %f' % (float(error_rates) / kf.n_folds)
	print '################'
	return (float(f1_scores) / kf.n_folds), (float(error_rates) / kf.n_folds)
		
def fusion_outputs(known_dataset, known_targets, train_index, test_index, fusion_algorithm, ids, current_svm):
	misclassified_ids = []
	combined_predictions = []
	y_test = []

	if fusion_algorithm == 'maj':
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids, current_svm)
		combined_predictions = majority_vote(predictions, y_test, accuracies)

	elif fusion_algorithm == 'wmaj':
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids, current_svm)
		combined_predictions = weighted_majority(predictions, y_test)

	elif fusion_algorithm == 'svm':
		y_test, predictions, combined_predictions, misclassified_ids = svm_fusion(known_dataset, known_targets, train_index, test_index, ids, current_svm)

	error = (float(sum((combined_predictions - y_test)**2)) / len(y_test))
	f1 = f1_score(combined_predictions, y_test)
	return error, f1, misclassified_ids

# Training and testing sets initially
# 2/3 are used to train the SVM and 1/3 is used to train(after the output is obtained) the fusion SVM
def svm_fusion(known_dataset, known_targets, train_index, test_index, ids, current_svm):
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

			model = current_svm.svm_for_features_fusion(svm_X_train, svm_Y_train)
			training_predictions.append(model.predict(fusion_X_train))
			predictions.append(model.predict(final_X_test))
			misclassified_ids += add_misclassified_ids(model, test_index, known_dataset[i], known_targets, ids)

		curr+=1
		if curr == 1:
			break

	training_pred_input = np.vstack(training_predictions).T
	fusion_model = current_svm.inner_svm(training_pred_input, fusion_Y_train)

	pred_input = np.vstack(predictions).T
	combined_predictions = fusion_model.predict(pred_input)

	return final_y_test, predictions, combined_predictions.tolist(), misclassified_ids

# called majority because it is used in both cases of majority_voting and weighted_majority voting.
def combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids, current_svm):
	misclassified_ids = []

	predictions = []
	accuracies = []
	y_train, y_test = known_targets[train_index], known_targets[test_index]
	for i in range(0, NR_THEMES):
		X_train, X_test = known_dataset[i][train_index], known_dataset[i][test_index]
		model = current_svm.svm_for_features_fusion(X_train, y_train)
		accuracy = model.score(X_test, y_test)
		y_pred = model.predict(X_test)
		predictions.append(y_pred)
		accuracies.append(accuracy)
		misclassified_ids += add_misclassified_ids(model, test_index, known_dataset[i], known_targets, ids)
	
	predictions = np.array((predictions[0], predictions[1], predictions[2]), dtype=float)
	return predictions, y_test, accuracies, misclassified_ids

# each fold has selected features
# after cv, choose the features that were selected at least N% of the repeated cross validation process (50%, 70%, 90%)
def features_cross_validation(known_dataset, known_targets, features, current_svm):
	kf = StratifiedKFold(known_targets, n_folds=10)
	error_rates = 0
	cv_features = []
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		
		error_rate, selected_features = selected_feature_one_fold(X_train, y_train, X_test, y_test, features, current_svm)
		error_rates += error_rate
		
		cv_features.append(selected_features)
	return cv_features	

# to be considered, a feature must appear at least 'percentage' of times in cv
def select_final_features_from_cv(cv_features, percentage):
	final_features = set()
	unique_features = set.union(*cv_features)
	nr_times = math.floor(percentage * len(cv_features))
	
	for feat in unique_features:
		if len([x for x in cv_features if feat in x]) >= nr_times:
			final_features.add(feat)

	return final_features	


def selected_feature_one_fold(X_train, y_train, X_test, y_test, features, current_svm):
	# train to get the selected features
	selected_features = feature_context(X_train, y_train, features)
	train_sf = SelectedFeatures(X_train, y_train, selected_features, features)
	test_sf = SelectedFeatures(X_test, y_test, selected_features, features)
	train_dataset_of_selected_features = train_sf.extract_data_from_selected_features()
	test_dataset_of_selected_features = test_sf.extract_data_from_selected_features()
	
	# check that there are the same number of examples (only features are removed)
	assert X_test.shape[0] ==  test_dataset_of_selected_features.shape[0]

	# test the selected features
	error_rate = one_fold_measures(train_dataset_of_selected_features, test_dataset_of_selected_features, y_train, y_test, current_svm)

	return error_rate, selected_features

def one_fold_measures(X_train, X_test, y_train, y_test, current_svm):
	model = current_svm.svm_subset_features(X_train, y_train)
	y_pred = model.predict(X_test)
	# error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	error_rate = f1_score(y_test, y_pred)
	return error_rate



class BestSVM:
	def __init__(self, c_subset, g_subset, c_fusion, g_fusion):
		self.c_subset = c_subset
		self.g_subset = g_subset
		self.c_fusion = c_fusion
		self.g_fusion = g_fusion

	# svm for fusion of outputs of the themes	
	def inner_svm(self, dataset, targets):
		model = SVC(class_weight='auto')
		model.fit(dataset, targets)
		return model

	# used for selecting the features	
	def svm_subset_features(self, dataset, targets):
		model = SVC(class_weight='auto', C=self.c_subset, gamma=self.g_subset)
		model.fit(dataset, targets)
		return model

	# used to train each theme	
	def svm_for_features_fusion(self, dataset, targets):
		model = SVC(class_weight='auto', C=self.c_fusion, gamma=self.g_fusion)
		model.fit(dataset, targets)
		return model

	def to_string(self):
		return 'c_subset %f, g_subset %f ||| c_fusion %f, g_fusion %f' % (self.c_subset, self.g_subset, self.c_fusion, self.g_fusion)	


def params():
	begin = 0.1
	end = 3
	C_range = np.arange(begin, end, 0.3)
	gamma_range = np.arange(begin, 1, 0.1)
	return C_range, gamma_range



if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	C_range, gamma_range = params()
	for pair in itertools.product(itertools.product(C_range,C_range), itertools.product(gamma_range,gamma_range)):
		c_subset = pair[0][0]
		g_subset = pair[1][0]
		c_fusion = pair[0][1]
		g_fusion = pair[1][1]
		current_svm = BestSVM(c_subset, g_subset, c_fusion, g_fusion)

		combined_dataset, targets = combine_data_from_feature_selection(targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD, current_svm)

		std = StandardizedData(targets)
		dataset = std.standardize_dataset(combined_dataset)  

		f1_maj, error_maj = cross_validation(dataset, targets, 'maj', ids, current_svm)

		if f1_maj == 0:
			continue

		f1_wmaj, error_wmaj = cross_validation(dataset, targets, 'wmaj', ids, current_svm)	
		f1_svm, error_svm = cross_validation(dataset, targets, 'svm', ids, current_svm)	
		
		if error_maj <= 0.3 or error_maj <= 0.3 or error_maj <= 0.3:
			with open("result.txt", "a") as myfile:	
				myfile.write('\n##############################\n')
			with open("result.txt", "a") as myfile:
				myfile.write(current_svm.to_string())
			with open("result.txt", "a") as myfile:	
				myfile.write('\nerror_maj %f' % error_maj)
			with open("result.txt", "a") as myfile:	
				myfile.write('\nerror_wmaj %f' % error_wmaj)
			with open("result.txt", "a") as myfile:	
				myfile.write('\nerror_svm %f' % error_svm)
    	
		print current_svm.to_string()
    	
    	


