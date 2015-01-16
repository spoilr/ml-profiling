print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from split_dataset import *
from feature_entropy import *
from join_attributes import *

import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

# each fold has selected features
# after cv, choose the features that were selected at least N% of the repeated cross validation process (50%, 80%, 90%)
def cross_validation(known_dataset, known_targets, features):
	kf = StratifiedKFold(known_targets, n_folds=5)
	error_rates = 0
	cv_features = []
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		
		# train to get the selected features
		selected_features = feature_context(X_train, y_train, features)
		train_dataset_of_selected_features = extract_data_from_selected_features(X_train, y_train, selected_features, features)
		test_dataset_of_selected_features = extract_data_from_selected_features(X_test, y_test, selected_features, features)
		
		# check that there are the same number of examples (only features are removed)
		assert X_test.shape[0] ==  test_dataset_of_selected_features.shape[0]

		# test the selected features
		error_rate = one_fold_measures(train_dataset_of_selected_features, test_dataset_of_selected_features, y_train, y_test)
		error_rates += error_rate
		
		cv_features.append(selected_features)

	print 'Final error rate %f' % (float(error_rates) / kf.n_folds)

	# for testing
	print cv_features
	for x in cv_features:
		print len(x)

def extract_data_from_selected_features(known_dataset, known_targets, selected_features, features):
	indices = extract_indices_of_selected_features(selected_features, features)
	dataset_of_selected_features = extract_examples_of_selected_features(known_dataset, known_targets, indices)
	return dataset_of_selected_features

def extract_indices_of_selected_features(selected_features, features):
		indices = []

		for x in selected_features:
			try:
				indices.append(features.index(x))
			except ValueError:
				print "Theme %s is not in the Features!" % x	

		return indices

def extract_examples_of_selected_features(known_dataset, known_targets, indices):
		num_rows = len(known_targets)
		examples = []

		for curr_row in range(num_rows):
			row = []
			for ind in indices:
				row.append(known_dataset[curr_row, ind])
			examples.append(row)	

		assert len(examples) == len(known_targets)
		return np.asarray(examples)	

def one_fold_measures(X_train, X_test, y_train, y_test):
	model = svm(X_train, y_train)
	print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	return error_rate	

def svm(dataset, targets):
	model = SVC()
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model	

if __name__ == "__main__":
	spreadsheet = Spreadsheet('/home/user/Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme(sys.argv[1])
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	
	# standardize dataset - Gaussian with zero mean and unit variance
	known_dataset_scaled = preprocessing.scale(known_dataset)
	known_targets = np.asarray(known_targets)

	cross_validation(known_dataset_scaled, known_targets, features)
