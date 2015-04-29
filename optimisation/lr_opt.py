import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
sys.path.insert(0, 'feature context/')
from load_data import *
from project_data import *
from parse_theme import *
from split_dataset import *
from feature_selection_before import select_features as sel_features
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from thematic_data_combined import *
from labels_fusion import *

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

import numpy as np

NR_THEMES = 3
themes = ['net', 'ill', 'ideo']
NR_FOLDS = 10

def cross_validation(known_dataset, known_targets, fusion_algorithm, ids, algorithm, c, prt=False, file_name=None, ind=False):
	misclassified_ids = []

	kf = StratifiedKFold(known_targets, n_folds=NR_FOLDS)
	f1_scores = 0
	error_rates = 0
	# cross validation
	for train_index, test_index in kf:
		error, f1 = fusion_outputs(known_dataset, known_targets, train_index, test_index, fusion_algorithm, ids, algorithm, c, ind)
		
		f1_scores += f1
		error_rates += error

	return (float(error_rates) / kf.n_folds)

def fusion_outputs(known_dataset, known_targets, train_index, test_index, fusion_algorithm, ids, algorithm, c, ind):
	misclassified_ids = []
	combined_predictions = []
	y_test = []

	predictions, y_test = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids, algorithm, c, ind)
	combined_predictions = weighted_majority(predictions, y_test)

	error = (float(sum((combined_predictions - y_test)**2)) / len(y_test))
	f1 = f1_score(combined_predictions, y_test)
	return error, f1

def combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids, algorithm, c, ind):
	predictions = []
	y_train, y_test = known_targets[train_index], known_targets[test_index]
	for i in range(0, NR_THEMES):
		X_train, X_test = known_dataset[i][train_index], known_dataset[i][test_index]

		model = algorithm(X_train, y_train, c)
		accuracy = model.score(X_test, y_test)
		y_pred = model.predict(X_test)
		predictions.append(y_pred)
	
	predictions = np.array((predictions[0], predictions[1], predictions[2]), dtype=float)
	return predictions, y_test

def lr(dataset, targets, c):
	model = LogisticRegression(class_weight='auto', C=c)
	model.fit(dataset, targets)
	return model

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	min_error = 70
	best_c = None

	C_range = np.arange(0.01, 70, 0.05)
	param_grid = dict(C=C_range)

	alg = raw_input("Enter type. Choose lr, lrft, singlelr, singlelrft")

	if alg == "singlelr":
		[dataset, features] = parse_theme('all')
		[known_dataset, known_targets, unk] = split_dataset(dataset, targets)


		cv = StratifiedShuffleSplit(known_targets, random_state=42)
		grid = GridSearchCV(LogisticRegression(class_weight='auto'), param_grid=param_grid, cv=cv, scoring='accuracy')
		grid.fit(known_dataset, known_targets)
		print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

	elif alg == "singlelrft":
		[dataset, features] = parse_theme('all')
		[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
		selected_features = sel_features(CV_PERCENTAGE_OCCURENCE_THRESHOLD)

		sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)
		known_dataset = sf.extract_data_from_selected_features()

		cv = StratifiedShuffleSplit(known_targets, random_state=42)
		grid = GridSearchCV(LogisticRegression(class_weight='auto'), param_grid=param_grid, cv=cv, scoring='accuracy')
		grid.fit(known_dataset, known_targets)
		print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

	elif alg == "lr":
		tdc = ThematicDataCombined(targets)
		dataset, targets = tdc.thematic_split() 
	
		for c in C_range:
			error = cross_validation(dataset, targets, 'wmaj', ids, lr, c)

			if error < min_error:
				min_error = error
				best_c = c

		print("The best parameters are %f with an error of %0.2f" % (best_c, min_error))		

	elif alg == "lrft":
		combined_dataset, targets = combine_data_from_feature_selection(targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD)

		for c in C_range:
			error = cross_validation(combined_dataset, targets, 'wmaj', ids, lr, c)
			if error < min_error:
				min_error = error
				best_c = c

		print("The best parameters are %f with an error of %0.2f" % (best_c, min_error))		

	else:
		print 'ERROR'	

	


