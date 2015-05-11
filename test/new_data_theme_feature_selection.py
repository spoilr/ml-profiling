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
from binary_classification_measures import measures
from opt_fusion_svm import combine_and_process_dataset
from svms import svm_selected_for_features_fusion
from svms import svm_selected_net
from svms import svm_selected_ill
from svms import svm_selected_ideo
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD	

import numpy as np
from sklearn.preprocessing import StandardScaler

def svm_vote(predictions, testing_targets):
	dataset, targets = combine_and_process_dataset()
	scaler = StandardScaler()
	dataset = scaler.fit_transform(dataset)
	model = inner_svm(dataset, targets)

	predictions = predictions.T
	predictions = scaler.transform(predictions)
	return model.predict(predictions)

def fusion(algorithm, training_data, training_targets, testing_data, testing_targets, fusion_algorithm, ind=False):
	models = []
	for i in range(NR_THEMES):

		if ind:
			if i == 0:
				model = svm_selected_net(training_data[i], training_targets)
			elif i == 1:
				model = svm_selected_ill(training_data[i], training_targets)
			elif i == 2:
				model = svm_selected_ideo(training_data[i], training_targets)
		else:
			model = algorithm(training_data[i], training_targets)
		models.append(model)

	predictions = []
	for i in range(NR_THEMES):
		y_pred = models[i].predict(testing_data[i])
		predictions.append(y_pred)
	predictions = np.array((predictions[0], predictions[1], predictions[2]), dtype=float)

	if fusion_algorithm == "maj":
		combined_predictions = majority_vote(predictions, testing_targets, [])
	elif fusion_algorithm == "wmaj":
		combined_predictions = weighted_majority(predictions, testing_targets)
	elif fusion_algorithm == "svm":	
		combined_predictions = svm_vote(predictions, testing_targets)
	else:
		print 'ERROR'	
	

	(hp, hr, hf), (cp, cr, cf) = measures(testing_targets, combined_predictions)
	error_rate = (float(sum((combined_predictions - testing_targets)**2)) / len(testing_targets))
	return error_rate, (hp, hr, hf), (cp, cr, cf)

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

	tech = raw_input("Enter algorithm. Choose between lr, dt, knn, svm")
	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")

	training_data, training_targets = combine_data_from_feature_selection(training_targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD)
	testing_data, testing_targets = combine_data_from_feature_selection_from_file(testing_targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD, addendum_data_file)

	net_scaler = StandardScaler()
	ill_scaler = StandardScaler()
	ideo_scaler = StandardScaler()

	if tech == 'lr':
		error_rate, (hp, hr, hf), (cp, cr, cf) = fusion(lr_feature_selection, training_data, training_targets, testing_data, testing_targets, fusion_algorithm)
		
	elif tech == 'dt':
		error_rate, (hp, hr, hf), (cp, cr, cf) = fusion(dt, training_data, training_targets, testing_data, testing_targets, fusion_algorithm)
		
	elif tech == 'knn':
		training_data[0] = net_scaler.fit_transform(training_data[0])
		training_data[1] =  ill_scaler.fit_transform(training_data[1])
		training_data[2] =  ideo_scaler.fit_transform(training_data[2])

		error_rate, (hp, hr, hf), (cp, cr, cf) = fusion(knn, training_data, training_targets, testing_data, testing_targets, fusion_algorithm)

	elif tech == 'svm':
		training_data[0] = net_scaler.fit_transform(training_data[0])
		training_data[1] =  ill_scaler.fit_transform(training_data[1])
		training_data[2] =  ideo_scaler.fit_transform(training_data[2])

		error_rate, (hp, hr, hf), (cp, cr, cf) = fusion(svm_selected_for_features_fusion, training_data, training_targets, testing_data, testing_targets, fusion_algorithm, ind=True)

	else:
		print 'ERROR technique'	

	print 'Final error %f' % error_rate
	print 'Final accuracy %f' % (1 - error_rate)

	print 'Highval precision %f' % hp
	print 'Highval recall %f' % hr
	print 'Highval f1 %f' % hf
	print 'Civil precision %f' % cp
	print 'Civil recall %f' % cr
	print 'Civil f1 %f' % cf


