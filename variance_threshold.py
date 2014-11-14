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
from sklearn.utils import validation
from sklearn.feature_selection import VarianceThreshold


# load the dataset
spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
data = Data(spreadsheet)

dataset = data.extract_illness_examples()
features = data.illness_features

model = VarianceThreshold(0.1)
a = model.fit_transform(dataset)


