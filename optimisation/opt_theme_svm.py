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
from sklearn.svm import SVC
from sklearn import preprocessing

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

	C_range = np.arange(0.1, 25, 0.3)
	gamma_range = np.arange(0.1, 25, 0.3)
	param_grid = dict(gamma=gamma_range, C=C_range)
	cv = StratifiedShuffleSplit(known_targets, random_state=42)
	grid = GridSearchCV(SVC(class_weight='auto'), param_grid=param_grid, cv=cv, scoring='accuracy')
	grid.fit(dataset, known_targets)
	print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
	