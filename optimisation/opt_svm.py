"""
Optimise parameters for a theme using all features from that theme
"""

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
sys.path.insert(0, 'results/')
from parse_theme import *
from split_dataset import *
from load_data import *
from project_data import *
from features_from_svm_selection import *
from selected_features import *

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn import preprocessing

def thematic_data_from_ft(theme, percentage, known_dataset, known_targets, features):
	selected_features = select_features(percentage, theme)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)
	return sf.extract_data_from_selected_features()

def select_features(percentage, theme):
	if theme == 'net' and percentage == 0.9:
		return net_90
	elif theme == 'net'and percentage == 0.7:
		return net_70
	elif theme == 'net'and percentage == 0.5:
		return net_50
	elif theme == 'ill' and percentage == 0.9:
		return ill_90
	elif theme == 'ill'and percentage == 0.7:
		return ill_70
	elif theme == 'ill'and percentage == 0.5:
		return ill_50
	elif theme == 'ideo' and percentage == 0.9:
		return ideo_90
	elif theme == 'ideo'and percentage == 0.7:
		return ideo_70
	elif theme == 'ideo'and percentage == 0.5:	
		return ideo_50
	elif theme == 'all' and percentage == 0.9:
		return single_features_90
	elif theme == 'all'and percentage == 0.7:
		return single_features_70
	elif theme == 'all'and percentage == 0.5:	
		return single_features_50	
	else:
		print 'ERROR in percentage - theme'

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	theme = raw_input("Theme.\n")
	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	known_targets = np.asarray(known_targets)


	percentage = float(raw_input("Percentage. 1, 0.9, 0.7 or 0.5\n"))

	if percentage < 1:
		known_dataset = thematic_data_from_ft(theme, percentage, known_dataset, known_targets, features)

	dataset = preprocessing.scale(known_dataset)	

	C_range = np.arange(0.1, 8, 0.3)
	gamma_range = np.arange(0.1, 8, 0.3)
	param_grid = dict(gamma=gamma_range, C=C_range)
	# cv = StratifiedShuffleSplit(known_targets, random_state=42)
	cv = StratifiedKFold(known_targets, n_folds=10)
	grid = GridSearchCV(SVC(class_weight='auto'), param_grid=param_grid, cv=cv, scoring='f1')
	grid.fit(dataset, known_targets)
	print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
	