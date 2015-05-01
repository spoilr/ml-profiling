"""
Optimise parameters for a theme using all features from that theme
"""

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
from parse_theme import *
from split_dataset import *
from load_data import *
from project_data import *

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn import preprocessing

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	theme = raw_input("Theme.\n")
	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	known_targets = np.asarray(known_targets)
	dataset = preprocessing.scale(known_dataset)

	C_range = np.arange(0.1, 10, 0.1)
	gamma_range = np.arange(0.1, 10, 0.1)
	param_grid = dict(gamma=gamma_range, C=C_range)
	# cv = StratifiedShuffleSplit(known_targets, random_state=42)
	cv = StratifiedKFold(known_targets, n_folds=10)
	grid = GridSearchCV(SVC(class_weight='auto'), param_grid=param_grid, cv=cv, scoring='f1')
	grid.fit(dataset, known_targets)
	print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
	