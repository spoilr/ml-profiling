"""
Optimize parameters for all features at once and for features split into categories.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

NR_THEMES = 3
themes = ['net', 'ill', 'ideo']
scores = ['recall', 'precision', 'accuracy']

class OptimizeParameters:

	def __init__(self, dataset, targets):
		self.dataset = dataset
		self.targets = targets

	def create_grid(self, score):
		begin = 10 ** (-5)
		end = 10 ** 2
		C_range = np.arange(begin, end, 0.3)
		gamma_range = np.arange(begin, end, 0.3)

		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = StratifiedKFold(y=self.targets, n_folds=5)
		grid = GridSearchCV(SVC(class_weight='auto'), param_grid=param_grid, scoring=score, cv=cv)

		return grid

	def all_optimize_parameters(self, score):	
		grid = self.create_grid(score)
		grid.fit(self.dataset, self.targets)
		print '######## %f ########' % score
		print("Best classifier: ", grid.best_estimator_)
		print("Best params: ", grid.best_params_)		
		

	def category_optimize_parameters(self, score, theme_index):
		grid = self.create_grid(score)
		categ_dataset = self.dataset[theme_index]	
		grid.fit(categ_dataset, self.targets)
		print '######## %f ########' % score
		print 'Category %s' % themes[theme_index]
		print("Best classifier: ", grid.best_estimator_)
		print("Best params: ", grid.best_params_)
			
			