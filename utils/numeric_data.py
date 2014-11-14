# Filter out the features that are not numeric

from load_data import *
import numpy as np

ignored_features = ['CoderID', 'Name', 'AliasList', 'DOB', 'AttackType', 'Target', 'TargetGroup']

def string_to_float(x):
    return float(x)

string_to_float = np.vectorize(string_to_float)

def numeric_features(features):
	return [x for x in features if x not in ignored_features]

def examples_from_numeric_features(examples, features):
	ignored_columns = []

	# get column index from the whole data set file 	 
	for x in ignored_features:
		try:
			ignored_columns.append(features.index(x))
		except ValueError:
			print "%s not in features to ignore" % x		

	numeric_examples = np.delete(examples, ignored_columns, 1) # remove columns from the data set

	return string_to_float(numeric_examples)

if __name__ == "__main__":
	spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	features = numeric_features(data.features)

	print 'nr features whose values are numeric %d \n' % len(features)
	print 'shape of new examples array'
	print examples_from_numeric_features(data.examples, data.features).shape