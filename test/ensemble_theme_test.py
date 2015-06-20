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
from save_output import *

import numpy as np
from sklearn.preprocessing import StandardScaler

def ensemble_theme(training_data, training_data_scaled, training_targets, testing_data, testing_data_scaled, testing_targets, fusion_algorithm):
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

def svm_vote(predictions):
	dataset, targets = combine_and_process_dataset()
	scaler = StandardScaler()
	dataset = scaler.fit_transform(dataset)
	model = inner_svm(dataset, targets)

	predictions = predictions.T
	predictions = scaler.transform(predictions)
	return model.predict(predictions)

def fusion(training_data, training_data_scaled, training_targets, testing_data, testing_data_scaled, testing_targets, fusion_algorithm):
	models_dt = []
	models_dt.append(dt(training_data[0], training_targets))
	models_dt.append(dt(training_data[1], training_targets))
	models_dt.append(dt(training_data[2], training_targets))

	models_knn = []
	models_knn.append(knn(training_data_scaled[0], training_targets))
	models_knn.append(knn(training_data_scaled[1], training_targets))
	models_knn.append(knn(training_data_scaled[2], training_targets))

	models_svm = []
	models_svm.append(svm_selected_net(training_data_scaled[0], training_targets))
	models_svm.append(svm_selected_ill(training_data_scaled[1], training_targets))
	models_svm.append(svm_selected_ideo(training_data_scaled[2], training_targets))

	predictions_dt = []
	predictions_knn = []
	predictions_svm = []
	for i in range(NR_THEMES):
		y_pred_dt = models_dt[i].predict(testing_data[i])
		predictions_dt.append(y_pred_dt)

		y_pred_knn = models_knn[i].predict(testing_data_scaled[i])
		predictions_knn.append(y_pred_knn)

		y_pred_svm = models_svm[i].predict(testing_data_scaled[i])
		predictions_svm.append(y_pred_svm)

	predictions_dt = np.array((predictions_dt[0], predictions_dt[1], predictions_dt[2]), dtype=float)
	predictions_knn = np.array((predictions_knn[0], predictions_knn[1], predictions_knn[2]), dtype=float)
	predictions_svm = np.array((predictions_svm[0], predictions_svm[1], predictions_svm[2]), dtype=float)	

	combined_predictions = []

	if fusion_algorithm == "maj":
		combined_predictions_dt = majority_vote(predictions_dt, testing_targets, [])
		combined_predictions_knn = majority_vote(predictions_knn, testing_targets, [])
		combined_predictions_svm = majority_vote(predictions_svm, testing_targets, [])
	elif fusion_algorithm == "wmaj":
		combined_predictions_dt = weighted_majority_theme('dt', predictions_dt)
		combined_predictions_knn = weighted_majority_theme('knn', predictions_knn)
		combined_predictions_svm = weighted_majority_theme('svm', predictions_svm)
	elif fusion_algorithm == "svm":	
		combined_predictions_dt = svm_vote(predictions_dt)
		combined_predictions_knn = svm_vote(predictions_knn)
		combined_predictions_svm = svm_vote(predictions_svm)
	else:
		print 'ERROR'		

	combined_predictions = []		
	for i in range(len(combined_predictions_dt)):
		data = Counter([combined_predictions_dt[i], combined_predictions_knn[i], combined_predictions_svm[i]])
		combined_predictions.append(data.most_common(1)[0][0])	
	
	print 'predictions DT ' + str(predictions_dt) 	
	print 'combined predictions DT ' + str(combined_predictions_dt) 	
	print 'predictions KNN ' + str(predictions_knn) 	
	print 'combined predictions KNN ' + str(combined_predictions_knn)
	print 'predictions SVM ' + str(predictions_svm) 	
	print 'combined predictions SVM ' + str(combined_predictions_svm)

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
	print 'WEIGHTS ' + theme + str(weights)
	combined_predictions = weigh(weights, predictions)
	return combined_predictions

if __name__ == "__main__":

	training_spreadsheet = Spreadsheet(project_data_file)
	training_data = Data(training_spreadsheet)
	training_targets = training_data.targets

	testing_spreadsheet = Spreadsheet(addendum_data_file, upsampling=False)
	testing_data = Data(testing_spreadsheet, upsampling=False)
	testing_targets = testing_data.targets

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")

	tdc = ThematicDataCombined(training_targets)
	training_data, training_targets = tdc.thematic_split() 	

	tdc = ThematicDataCombined(testing_targets)
	testing_data, testing_targets = tdc.thematic_split_from_file(addendum_data_file) 	

	
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

	file_name = "ensemble_theme.txt"
	for i in range(100):
		error_rate, (hp, hr, hf), (cp, cr, cf) = ensemble_theme(training_data, training_data_scaled, training_targets, testing_data, testing_data_scaled, testing_targets, fusion_algorithm)
		save_output(file_name, error_rate, hp, hr, hf, cp, cr, cf, 1)


