import sys
sys.path.insert(0, 'classification/')
from parameters import TOP_FEATURES_PERCENTAGE_THRESHOLD
sys.path.insert(0, 'feature context/')
from join_attributes import feature_context

import math as math
import operator
import numpy as np


def feature_context_test():
	# dataset = np.array([[1,0,0,1,1], [1,0,0,0,0], [1,0,1,1,1]])
	# targets = [1, 0, 1]
	# features = ['a', 'b', 'c', 'd', 'e']
	features = ["out", "temp", "humid", "wind"]
	targets = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]
	dataset = np.array([[1,1,1,1], [1,1,1,2], [2,1,1,1], [3,3,1,1], [3,2,2,1], [3,2,2,2,], [2,2,2,2], [1,3,1,1], [1,2,2,1], [3,3,2,1], [1,3,2,2], [2,3,1,2], [2,1,2,1], [3,3,1,2]])
	print feature_context(dataset, targets, features)

feature_context_test()