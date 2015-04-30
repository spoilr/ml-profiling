"""
SVM C-Support Vector Classification
Single SVM
Feature selection is applied before
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
sys.path.insert(0, 'feature context/')
from load_data import *
from project_data import *
from parse_theme import *
from feature_selection_before import *
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from cv import single_svm_fs_one_fold_measures

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn import preprocessing

import numpy as np

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		known_targets = np.array(targets)

		selected_features = select_features(CV_PERCENTAGE_OCCURENCE_THRESHOLD)
		sf = SelectedFeatures(dataset, known_targets, selected_features, features)
		dataset = sf.extract_data_from_selected_features()

		dataset = preprocessing.scale(dataset)

		C_range = np.arange(0.1, 16, 0.1)
		gamma_range = np.arange(0.1, 16, 0.1)
		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = StratifiedShuffleSplit(known_targets, random_state=42)
		grid = GridSearchCV(SVC(class_weight='auto'), param_grid=param_grid, cv=cv, scoring='accuracy')
		grid.fit(dataset, known_targets)
		print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
		
	except IndexError:
		print "Error!! Pass 'all' as argument"
