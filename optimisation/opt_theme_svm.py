"""
Optimise parameters for a theme given percentage using GridSearchCV
"""

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import *
from parse_theme import *
from split_dataset import *
from selected_features import *
from thematic_data_combined import select_features

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn import preprocessing

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	theme = raw_input("Theme.\n")
	percentage = float(raw_input("Percentage as float.\n"))

	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
	known_targets = np.asarray(known_targets)
	
	selected_features = select_features(percentage, theme)
	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)
	dataset = sf.extract_data_from_selected_features()

	dataset = preprocessing.scale(dataset)

	C_range = np.arange(0.1, 9, 0.1)
	gamma_range = np.arange(0.1, 9, 0.1)
	param_grid = dict(gamma=gamma_range, C=C_range)
	# cv = StratifiedShuffleSplit(known_targets, random_state=42)
	cv = StratifiedKFold(known_targets, n_folds=10)
	grid = GridSearchCV(SVC(class_weight='auto'), param_grid=param_grid, cv=cv, scoring='f1')
	grid.fit(dataset, known_targets)
	print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

	classifiers = []
	for C in C_range:
	    for gamma in gamma_range:
	        clf = SVC(C=C, gamma=gamma)
	        clf.fit(dataset, known_targets)
	        classifiers.append((C, gamma, clf))

	##############################################################################
	# visualization
	
	scores = [x[1] for x in grid.grid_scores_]
	scores = np.array(scores).reshape(len(C_range), len(gamma_range))

	plt.figure(figsize=(8, 6))
	plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
	plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
	           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
	plt.xlabel('gamma')
	plt.ylabel('C')
	plt.colorbar()
	plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
	plt.yticks(np.arange(len(C_range)), C_range)
	plt.title('Validation accuracy')
	plt.show()        
	