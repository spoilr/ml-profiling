"""
================================
Supervised Variance Threshold
================================
"""

''' Feature selector that removes all low-variance features.
This feature selection algorithm looks only at the features (X), not the desired outputs (y), 
and can thus be used for unsupervised learning. '''

print(__doc__)

import sys
sys.path.insert(0, './utils/')
from load_data import *

import operator
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.utils import validation
from sklearn.feature_selection import VarianceThreshold


# load the dataset
spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
data = Data(spreadsheet)
targets = data.targets

print len(targets)
print len([x for x in targets if x == 1])
print len([x for x in targets if x == 2])



dataset = data.extract_illness_examples()
features = data.illness_features

model = VarianceThreshold(0.2)
a = model.fit_transform(dataset, targets)


# variance per feature
feature_variance = np.var(dataset, axis = 0)
assert len(feature_variance) == len(dataset[0])

# create dictionary of feature and variance
variance_per_feature = {}
for i in range(0, len(feature_variance)):
	variance_per_feature[features[i]] = feature_variance[i]

decreasing_variance_per_feature = OrderedDict(sorted(variance_per_feature.items(), key=lambda t: t[1], reverse = True))

# create tuples of feature and variance
variances = sorted(variance_per_feature.items(), key=operator.itemgetter(1), reverse=True)

#######################################################
# plot feature variance
N = len(variances)
x = np.arange(1, N+1)
y = [num for (s, num) in variances]
labels = [s for (s, num) in variances]
width = 1
bar1 = plt.bar(x, y, width, color="y")
plt.ylabel('Variance')
plt.xticks(x + width/2.0, labels, rotation=45)
plt.gcf().subplots_adjust(bottom=0.25)
plt.show()