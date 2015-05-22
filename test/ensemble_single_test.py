import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import *
from parse_theme import *
from standardized_data import *
from split_dataset import *
from cv import ensemble_one_fold_measures
from cv import dt
from cv import knn
from svms import svm_all_vars
from replace_missing_values import *

import numpy as np
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

	training_spreadsheet = Spreadsheet(project_data_file)
	training_data = Data(training_spreadsheet)
	training_targets = training_data.targets

	testing_spreadsheet = Spreadsheet(addendum_data_file, upsampling=False)
	testing_data = Data(testing_spreadsheet, upsampling=False)
	testing_targets = testing_data.targets

	[training_data, features] = parse_theme('all')
	[testing_data, feats] = parse_theme_from_file('all', addendum_data_file)
	assert features == feats

	[training_data, training_targets, unk] = split_dataset(training_data, training_targets)
		
	# standardize dataset - Gaussian with zero mean and unit variance
	scaler = StandardScaler()

	testing_data = replace_missings(testing_data)

	training_data_scaled = scaler.fit_transform(training_data)
	testing_data_scaled = scaler.transform(testing_data)

	error_rate, f1, (hp, hr, hf), (cp, cr, cf) = ensemble_one_fold_measures(training_data, testing_data, training_data_scaled, testing_data_scaled, np.array(training_targets), np.array(testing_targets), dt, knn, svm_all_vars)

	print 'Final error %f' % error_rate
	print 'Final accuracy %f' % (1 - error_rate)

	print 'Highval precision %f' % hp
	print 'Highval recall %f' % hr
	print 'Highval f1 %f' % hf
	print 'Civil precision %f' % cp
	print 'Civil recall %f' % cr
	print 'Civil f1 %f' % cf
