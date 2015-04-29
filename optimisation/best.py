"""
Decision Tree Classification
Combine DT for themes
Feature selection is applied before
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import *
from fusion import cv10
from fusion import dt
from fusion import lr_feature_selection
from fusion import knn
from thematic_data_combined import combine_data_from_feature_selection
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from svms import svm_selected_for_features_fusion

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	combined_dataset, targets = combine_data_from_feature_selection(targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD)

	alg = raw_input("Enter algorithm. Choose lr, dt, knn, svm")
	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")

	for i in range(1000):
		if alg == "lr":
			cv10(combined_dataset, targets, fusion_algorithm, ids, lr_feature_selection, prt=True, file_name="best_lr.txt")
		elif alg == "dt":
			cv10(combined_dataset, targets, fusion_algorithm, ids, dt, prt=True, file_name="best_dt.txt")
		elif alg == "knn":
			cv10(combined_dataset, targets, fusion_algorithm, ids, knn, prt=True, file_name="best_knn.txt")
		elif alg == "svm":

			std = StandardizedData(targets)
			dataset = std.standardize_dataset(combined_dataset)  
			cv10(dataset, targets, fusion_algorithm, ids, svm_selected_for_features_fusion, ind=True, prt=True, file_name="best_svm.txt")	
		else:
			print 'ERROR'	

	
		
