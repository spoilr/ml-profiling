import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn import preprocessing

net 	= np.array([2,  1,  2,  2,  1,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  1,  1,  2,  2,  2, 1,  2,  2,  2,  2,  2,  2,  2,  2,  2, 1,  2,  2,  2,  1,  2,  2,  2,  2,  1, 2,  2,  2,  1,  2,  2,  2,  2,  2, 1,  1,  2,  1,  2,  2,  2,  2,  2, 2,  1,  1,  2,  1,  2,  1,  2,  2, 1,  1,  1,  1,  2,  2,  2,  1,  1, 2,  1,  2,  2,  2,  2,  2,  2, 1,  2,  2,  2,  2,  2,  2,  1])
ill 	= np.array([2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2])
ideo 	= np.array([2,  2,  2,  2,  1,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  1,  2,  1,  2,  2,  2,  2,  1, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  1,  2,  2,  2,  2,  2, 2,  1,  2,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 1,  1,  2,  2,  2,  2,  2,  2, 2,  2,  2,  2,  1,  2,  2,  2])
targets = np.array([1,1,1,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,2,2,2,2,2,2,1,1,2,2,2,2,2,2])

if __name__ == "__main__":

	data = []
	for i in range(len(targets)):
		data.append([float(net[i]), float(ill[i]), float(ideo[i])])

	dataset = preprocessing.scale(data)

	C_range = np.arange(0.1, 16, 0.05)
	gamma_range = np.arange(0.1, 16, 0.05)
	param_grid = dict(gamma=gamma_range, C=C_range)
	cv = StratifiedShuffleSplit(targets, random_state=42)
	grid = GridSearchCV(SVC(class_weight='auto'), param_grid=param_grid, cv=cv, scoring='accuracy')
	grid.fit(dataset, targets)
	print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
