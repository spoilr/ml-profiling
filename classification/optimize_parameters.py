"""
Optimize parameters for all features at once and for features split into categories.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

NR_THEMES = 3

class OptimizeParameters:

	def __init__(self, dataset, targets):
		self.dataset = dataset
		self.targets = targets

	def create_grid(self):
		C_range = 0.1 * np.array(range(1, 50))
		gamma_range = 0.1 * np.array(range(1, 50))
		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = StratifiedKFold(y=self.targets, n_folds=3)
		grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=cv)
		return grid

	def all_optimize_parameters(self):	
		grid = self.create_grid()
		grid.fit(self.dataset, self.targets)
		print("Best classifier: ", grid.best_estimator_)

	def category_optimize_parameters(self):
		grid = self.create_grid()
		for i in range(0, NR_THEMES):
			categ_dataset = self.dataset[i]	
			grid.fit(categ_dataset, self.targets)
			print("Best classifier: ", grid.best_estimator_)