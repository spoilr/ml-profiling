non_binary_vars = ['RelatStat', 'ParRelatStat', 'Education', 'OccCat', 'MilExp', 'CrimCon', 'Ideology', 'Religion', 
					'IdeoChangeInt', 'ReligChangeInt', 'LifeAspectChange', 'Funds', 'TargetTyp', 'LocationNature', 'MultiEventTarget']

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import *

def extract_indices_of_non_binary_vars(features):
	non_binary_indices = []

	for x in non_binary_vars:
		try:
			non_binary_indices.append(features.index(x))
		except ValueError:
			print "Theme %s is not in the Main Features!" % x	

	return non_binary_indices

def extract_binaries(feature_non_binary_data, val):
	for x in feature_non_binary_data:
		if x == val:
			print 1
		else:
			print 0			

def extract_binaries_for_feature(non_binary_data, i):
	print 'Feature %s' % non_binary_vars[i]
	feature_non_binary_data = non_binary_data[:,i]
	vals = set(feature_non_binary_data)
	for val in vals:
		print 'Val %f' % val
		extract_binaries(feature_non_binary_data, val)

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	features = data.features

	non_binary_indices = extract_indices_of_non_binary_vars(features)
	non_binary_data = data.extract_examples_with_features_from_indices(non_binary_indices)

	for i in range(len(non_binary_indices)):
		extract_binaries_for_feature(non_binary_data, i)
