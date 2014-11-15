"""
============================
Variance Threshold
============================
"""

''' Feature selector that removes all low-variance features.
This feature selection algorithm looks only at the features (X), not the desired outputs (y), 
and can thus be used for unsupervised learning. '''

print(__doc__)

from utils.load_data import *
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.utils import validation
from sklearn.feature_selection import VarianceThreshold


# load the dataset
spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
data = Data(spreadsheet)

dataset = data.extract_illness_examples()
features = data.illness_features

model = VarianceThreshold(0.2)
a = model.fit_transform(dataset)


# variance per feature
feature_variance = np.var(dataset, axis = 0)
assert len(feature_variance) == len(dataset[0])

# create dictionary of feature and variance
variance_per_feature = {}
for i in range(0, len(feature_variance)):
	variance_per_feature[features[i]] = feature_variance[i]

print OrderedDict(sorted(variance_per_feature.items(), key=lambda t: t[1], reverse = True))
