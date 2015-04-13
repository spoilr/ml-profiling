print(__doc__)


import sys
sys.path.insert(0, 'utils/')
from load_data import *
from knn_fusion import *
from selected_features import *
sys.path.insert(0, 'feature context/')
from thematic_data_combined import combine_data_from_feature_selection
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	combined_dataset, targets = combine_data_from_feature_selection(targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD)

	std = StandardizedData(targets)
	dataset = std.standardize_dataset(combined_dataset)  

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	cross_validation(dataset, targets, fusion_algorithm, ids)

	