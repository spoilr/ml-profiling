import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import *
from parse_theme import *
from feature_selection_before import *
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from cv import ensemble_one_fold_measures
from cv import dt
from cv import knn
from svms import svm_selected_vars
from features_from_svm_selection import single_features_90
from replace_missing_values import *
from save_output import *

import numpy as np
from sklearn.preprocessing import StandardScaler

def ensemble_single_feature_selection(training_data, testing_data, training_data_scaled, testing_data_scaled, training_targets, testing_targets, dt, knn, svm_selected_vars):
	error_rate, f1, (hp, hr, hf), (cp, cr, cf) = ensemble_one_fold_measures(training_data, testing_data, training_data_scaled, testing_data_scaled, np.array(training_targets), np.array(testing_targets), dt, knn, svm_selected_vars)


	print 'Final error %f' % error_rate
	print 'Final accuracy %f' % (1 - error_rate)

	print 'Highval precision %f' % hp
	print 'Highval recall %f' % hr
	print 'Highval f1 %f' % hf
	print 'Civil precision %f' % cp
	print 'Civil recall %f' % cr
	print 'Civil f1 %f' % cf

	return error_rate, f1, (hp, hr, hf), (cp, cr, cf)

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
	selected_features = single_features_90
	sf = SelectedFeatures(training_data, training_targets, selected_features, features)
	training_data = sf.extract_data_from_selected_features()

	sf = SelectedFeatures(testing_data, testing_targets, selected_features, features)
	testing_data = sf.extract_data_from_selected_features()
		
	# standardize dataset - Gaussian with zero mean and unit variance
	scaler = StandardScaler()

	testing_data = replace_missings(testing_data)
	training_data_scaled = scaler.fit_transform(training_data)
	testing_data_scaled = scaler.transform(testing_data)
	
	file_name = "ensemble_single_feature_selection.txt"
	for i in range(100):
		error_rate, f1, (hp, hr, hf), (cp, cr, cf) = ensemble_single_feature_selection(training_data, testing_data, training_data_scaled, testing_data_scaled, training_targets, testing_targets, dt, knn, svm_selected_vars)
		save_output(file_name, error_rate, hp, hr, hf, cp, cr, cf, 1)

