"""
Ensemble Classification using DT, KNN, SVM
Combine for themes
"""

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import *
from fusion import cv10_ensemble
from fusion import dt
from fusion import knn
from svms import svm_for_features_fusion
from thematic_data_combined import *
from standardized_data import *

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")

	tdc = ThematicDataCombined(targets)
	dataset, targets = tdc.thematic_split()
	std = StandardizedData(targets)
	dataset_scaled, targets = std.thematic_split_and_standardize_dataset() 

	file_name = fusion_algorithm + "_ensemble.txt"
	for i in range(100):
		cv10_ensemble(dataset, targets, dataset_scaled, dt, knn, svm_for_features_fusion, fusion_algorithm, ids, prt=True, file_name=file_name)
