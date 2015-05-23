import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
sys.path.insert(0, 'optimisation/')
from load_data import *
from project_data import *
from parse_theme import *
from standardized_data import *
from fusion import lr_feature_selection
from fusion import dt
from fusion import knn
from fusion import inner_svm
from labels_fusion import *
from thematic_data_combined import *
from weights import *
from binary_classification_measures import measures
from opt_fusion_svm import combine_and_process_dataset
from svms import svm_selected_for_features_fusion
from svms import svm_selected_net
from svms import svm_selected_ill
from svms import svm_selected_ideo
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD	
from replace_missing_values import *
from ensemble_theme_test import svm_vote
from ensemble_theme_test import fusion
from ensemble_theme_test import weighted_majority_theme
from save_output import *

import numpy as np
from sklearn.preprocessing import StandardScaler

def ensemble_theme_feature_selection(training_data, training_data_scaled, training_targets, testing_data, testing_data_scaled, testing_targets, fusion_algorithm):
	error_rate, (hp, hr, hf), (cp, cr, cf) = fusion(training_data, training_data_scaled, training_targets, testing_data, testing_data_scaled, testing_targets, fusion_algorithm)
	
	print 'Final error %f' % error_rate
	print 'Final accuracy %f' % (1 - error_rate)

	print 'Highval precision %f' % hp
	print 'Highval recall %f' % hr
	print 'Highval f1 %f' % hf
	print 'Civil precision %f' % cp
	print 'Civil recall %f' % cr
	print 'Civil f1 %f' % cf

	return error_rate, (hp, hr, hf), (cp, cr, cf)

if __name__ == "__main__":

	training_spreadsheet = Spreadsheet(project_data_file)
	training_data = Data(training_spreadsheet)
	training_targets = training_data.targets

	testing_spreadsheet = Spreadsheet(addendum_data_file, upsampling=False)
	testing_data = Data(testing_spreadsheet, upsampling=False)
	testing_targets = testing_data.targets

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")

	training_data, training_targets = combine_data_from_feature_selection(training_targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD)
	testing_data, testing_targets = combine_data_from_feature_selection_from_file(testing_targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD, addendum_data_file)

	net_scaler = StandardScaler()
	ill_scaler = StandardScaler()
	ideo_scaler = StandardScaler()

	testing_data = replace_missings_thematic(testing_data)

	training_data_scaled = []
	training_data_scaled.append(net_scaler.fit_transform(training_data[0]))
	training_data_scaled.append(ill_scaler.fit_transform(training_data[1]))
	training_data_scaled.append(ideo_scaler.fit_transform(training_data[2]))

	testing_data_scaled = []
	testing_data_scaled.append(net_scaler.transform(testing_data[0]))
	testing_data_scaled.append(ill_scaler.transform(testing_data[1]))
	testing_data_scaled.append(ideo_scaler.transform(testing_data[2]))

	file_name = "ensemble_theme_feature_selection.txt"
	for i in range(100):
		error_rate, (hp, hr, hf), (cp, cr, cf) = ensemble_theme_feature_selection(training_data, training_data_scaled, training_targets, testing_data, testing_data_scaled, testing_targets, fusion_algorithm)
		save_output(file_name, error_rate, hp, hr, hf, cp, cr, cf, 1)


