import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
sys.path.insert(0, 'optimisation/')
sys.path.insert(0, 'results/')
from load_data import *
from project_data import *
from parse_theme import *
from standardized_data import *
from fusion import lr
from fusion import dt
from fusion import knn
from fusion import inner_svm
from labels_fusion import *
from thematic_data_combined import *
from weights import *
from binary_classification_measures import measures
from opt_fusion_svm import combine_and_process_dataset
from svms import svm_for_features_fusion
from svms import svm_selected_net
from svms import svm_selected_ill
from svms import svm_selected_ideo
from replace_missing_values import *
from save_output import save_output

import numpy as np
from sklearn.preprocessing import StandardScaler

def new_data_theme(training_data, training_targets, testing_data, testing_targets, tech, fusion_algorithm):
	tdc = ThematicDataCombined(training_targets)
	training_data, training_targets = tdc.thematic_split() 	

	tdc = ThematicDataCombined(testing_targets)
	testing_data, testing_targets = tdc.thematic_split_from_file(addendum_data_file) 	

	
	net_scaler = StandardScaler()
	ill_scaler = StandardScaler()
	ideo_scaler = StandardScaler()

	testing_data = replace_missings_thematic(testing_data)

	if tech == 'lr':
		error_rate, (hp, hr, hf), (cp, cr, cf) = fusion('lr', lr, training_data, training_targets, testing_data, testing_targets, fusion_algorithm)
		
	elif tech == 'dt':
		error_rate, (hp, hr, hf), (cp, cr, cf) = fusion('dt', dt, training_data, training_targets, testing_data, testing_targets, fusion_algorithm)
		
	elif tech == 'knn':
		training_data[0] = net_scaler.fit_transform(training_data[0])
		training_data[1] =  ill_scaler.fit_transform(training_data[1])
		training_data[2] =  ideo_scaler.fit_transform(training_data[2])

		# testing_data[0] = net_scaler.transform(testing_data[0])
		# testing_data[1] =  ill_scaler.transform(testing_data[1])
		# testing_data[2] =  ideo_scaler.transform(testing_data[2])

		error_rate, (hp, hr, hf), (cp, cr, cf) = fusion('knn', knn, training_data, training_targets, testing_data, testing_targets, fusion_algorithm)

	elif tech == 'svm':
		training_data[0] = net_scaler.fit_transform(training_data[0])
		training_data[1] =  ill_scaler.fit_transform(training_data[1])
		training_data[2] =  ideo_scaler.fit_transform(training_data[2])

		testing_data[0] = net_scaler.transform(testing_data[0])
		testing_data[1] =  ill_scaler.transform(testing_data[1])
		testing_data[2] =  ideo_scaler.transform(testing_data[2])

		error_rate, (hp, hr, hf), (cp, cr, cf) = fusion('svm', svm_for_features_fusion, training_data, training_targets, testing_data, testing_targets, fusion_algorithm, ind=True)

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

	return error_rate, (hp, hr, hf), (cp, cr, cf)

def svm_vote(predictions, testing_targets):
	dataset, targets = combine_and_process_dataset()
	scaler = StandardScaler()
	dataset = scaler.fit_transform(dataset)
	model = inner_svm(dataset, targets)

	predictions = predictions.T
	predictions = scaler.transform(predictions)
	return model.predict(predictions)

def fusion(theme, algorithm, training_data, training_targets, testing_data, testing_targets, fusion_algorithm, ind=False):
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
		combined_predictions = weighted_majority_theme(theme, predictions)
	elif fusion_algorithm == "svm":	
		combined_predictions = svm_vote(predictions, testing_targets)
	else:
		print 'ERROR'	
	
	print 'PRED ' + str(combined_predictions)
	print 'TEST ' + str(testing_targets)	
	
	(hp, hr, hf), (cp, cr, cf) = measures(testing_targets, combined_predictions)
	error_rate = (float(sum((combined_predictions - testing_targets)**2)) / len(testing_targets))
	return error_rate, (hp, hr, hf), (cp, cr, cf)

def weighted_majority_theme(theme, predictions):
	if theme == 'lr':
		weights = get_lr_weights()
	elif theme == 'dt':
		weights = get_dt_weights()
	elif theme == 'knn':
		weights = get_knn_weights()		
	elif theme == 'svm':
		weights = get_svm_weights()	
	combined_predictions = weigh(weights, predictions)
	return combined_predictions

if __name__ == "__main__":

	training_spreadsheet = Spreadsheet(project_data_file)
	training_data = Data(training_spreadsheet)
	training_targets = training_data.targets

	testing_spreadsheet = Spreadsheet(addendum_data_file, upsampling=False)
	testing_data = Data(testing_spreadsheet, upsampling=False)
	testing_targets = testing_data.targets

	tech = raw_input("Enter algorithm. Choose between lr, dt, knn, svm")
	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	file_name = "new_theme_" + tech + fusion_algorithm + ".txt"
	for i in range(100):
		error_rate, (hp, hr, hf), (cp, cr, cf) = new_data_theme(training_data, training_targets, testing_data, testing_targets, tech, fusion_algorithm)
		save_output(file_name, error_rate, hp, hr, hf, cp, cr, cf, 1)
	


