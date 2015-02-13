import numpy as np
import itertools
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
import sys
sys.path.insert(0, 'utils/')
from standardized_data import *
from svm_feature_selection import combine_data_from_feature_selection
from selected_features import *
sys.path.insert(0, 'feature context/')
from feature_selection_cv import features_cross_validation
from feature_selection_cv import select_final_features_from_cv

def params():
	begin = 10 ** (-3)
	end = 50
	C_range = np.arange(begin, end, 0.1)
	gamma_range = np.arange(begin, end, 0.1)
	return C_range, gamma_range

# single svm - no thematic split
def cross_validation_single(known_dataset, known_targets, model):
	kf = StratifiedKFold(known_targets, n_folds=10)
	error_rates = 0
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]

		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		# error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
		error_rate = f1_score(y_test, y_pred)

		error_rates += error_rate

	return float(error_rates) / kf.n_folds

def opt_params_single(known_dataset, known_targets):
	c_param = -1
	g_param = -1
	maxF1 = 0.0
	C_range, gamma_range = params()
	for pair in itertools.product(C_range, gamma_range):
		c = pair[0]
		g = pair[1]
		model = SVC(class_weight='auto', C=c, gamma=g)
		f1 = cross_validation_single(known_dataset, known_targets, model)
		if f1 > maxF1:
			maxF1 = f1
			c_param = c
			g_param = g

	print '##### BEST PARAMS #####: C %f and g %f with f1 %f' % (c_param, g_param, maxF1)

def cross_validation_theme(known_dataset, known_targets, theme_index, model):
	kf = StratifiedKFold(known_targets, n_folds=10)
	error_rates = 0
	# cross validation
	for train_index, test_index in kf:
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		X_train, X_test = known_dataset[theme_index][train_index], known_dataset[theme_index][test_index]

		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		# error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
		error_rate = f1_score(y_test, y_pred)

		error_rates += error_rate

	return float(error_rates) / kf.n_folds

def opt_params_combined(known_dataset, known_targets, theme_index):
	c_param = -1
	g_param = -1
	maxF1 = 0.0
	C_range, gamma_range = params()
	for pair in itertools.product(C_range, gamma_range):
		c = pair[0]
		g = pair[1]
		model = SVC(class_weight='auto', C=c, gamma=g)
		f1 = cross_validation_theme(known_dataset, known_targets, theme_index, model)
		if f1 > maxF1:
			maxF1 = f1
			c_param = c
			g_param = g

	print '##### BEST PARAMS #####: C %f and g %f with f1 %f' % (c_param, g_param, maxF1)

def main_svm():
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets

	std = StandardizedData(targets)
	dataset, targets = std.thematic_split_and_standardize_dataset()

	theme_index = int(raw_input('Theme index? 0, 1 or 2'))
	opt_params_combined(dataset, targets, theme_index)

def main_svm_feature_selection():
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	orig_targets = data.targets

	combined_dataset, targets = combine_data_from_feature_selection(orig_targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD)

	std = StandardizedData(targets)
	dataset = std.standardize_dataset(combined_dataset)  

	theme_index = int(raw_input('Theme index? 0, 1 or 2'))
	opt_params_combined(dataset, targets, theme_index)
		

def main_single_svm():
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme('all')

	std = StandardizedData(targets, dataset)
	known_dataset_scaled, known_targets = std.split_and_standardize_dataset()

	opt_params_single(known_dataset_scaled, known_targets)

def main_single_svm_feature_selection():
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme('all')

	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
		
	known_targets = np.asarray(known_targets)

	cv_features = features_cross_validation(known_dataset, known_targets, features)
	selected_features = select_final_features_from_cv(cv_features, CV_PERCENTAGE_OCCURENCE_THRESHOLD)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)
	combined_dataset = sf.extract_data_from_selected_features()

	std = StandardizedData(known_targets)
	combined_dataset = std.standardize_dataset(combined_dataset)  

	opt_params_single(np.array(combined_dataset), known_targets)

		
if __name__ == "__main__":
	case = raw_input("Choose case: svm, svmf, singlesvm, singlesvmf")
	if case == 'svm':
		main_svm()
	elif case == 'svmf':
		main_svm_feature_selection()
	elif case == 'singlesvm':
		main_single_svm()
	elif case == 'singlesvmf':
		main_single_svm_feature_selection()
	else:
		print 'ERROR CASE'				

