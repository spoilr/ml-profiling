"""
Ensemble Classification using DT, KNN, SVM
Combine for themes Feature selection
"""

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
from load_data import *
from project_data import *
from fusion import cv10_ensemble
from fusion import dt
from fusion import knn
from svms import svm_selected_for_features_fusion
from standardized_data import *
from thematic_data_combined import combine_data_from_feature_selection
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")

	combined_dataset, targets = combine_data_from_feature_selection(targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD)
	std = StandardizedData(targets)
	combined_dataset_scaled = std.standardize_dataset(combined_dataset)  
	
	file_name = fusion_algorithm + "_ensemble_ft.txt"
	for i in range(100):
		cv10_ensemble(combined_dataset, targets, combined_dataset_scaled, dt, knn, svm_selected_for_features_fusion, fusion_algorithm, ids, prt=True, file_name=file_name)