# Filter out the features that are not numeric

from load_data import *
import numpy as np

ignored_features = ['CoderID', 'Name', 'AliasList', 'DOB', 'AttackType', 'Target', 'TargetGroup']

spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
data = Data(spreadsheet)

def numeric_features():
	return [x for x in data.features if x not in ignored_features]

def examples_from_numeric_features():
	ignored_columns = [data.features.index(x) for x in ignored_features] # get column index from the whole data set file 	 
	return np.delete(data.examples, ignored_columns, 1) # remove columns from the data set
