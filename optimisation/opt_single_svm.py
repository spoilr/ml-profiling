"""
SVM C-Support Vector Classification
Single SVM
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import *
from parse_theme import *
from standardized_data import *

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn import preprocessing

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		std = StandardizedData(targets, dataset)
		known_dataset_scaled, known_targets = std.split_and_standardize_dataset()

		C_range = np.arange(0.1, 25, 0.05)
		gamma_range = np.arange(0.1, 25, 0.05)
		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = StratifiedShuffleSplit(known_targets, random_state=42)
		grid = GridSearchCV(SVC(class_weight='auto'), param_grid=param_grid, cv=cv, scoring='accuracy')
		grid.fit(known_dataset_scaled, known_targets)
		print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
		
	except IndexError:
		print "Error!! Pass 'all' as argument"
