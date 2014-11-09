# Feature Importance
from load_data import *
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomTreesEmbedding

# load the dataset
spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
data = Data(spreadsheet)

dataset = data.extract_illness_examples()


# fit an Extra Trees model to the data
model = RandomTreesEmbedding()
model.fit(dataset)

# display the relative importance of each attribute
print(model.feature_importances_)