import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import *
from parse_theme import *
from feature_selection_before import *
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from cv import dt_one_fold_measures
from cv import lr_one_fold_measures_feature_selection
from cv import knn_one_fold_measures
from cv import single_svm_fs_one_fold_measures

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		alg = raw_input("Enter algorithm. Choose lr, dt, knn, svm")

		for i in range(100):
			if alg == "lr":
				feature_selection_before(features, targets, dataset, CV_PERCENTAGE_OCCURENCE_THRESHOLD, ids, lr_one_fold_measures_feature_selection, prt=True, file_name="best_single_lr.txt")
			elif alg == "dt":
				feature_selection_before(features, targets, dataset, CV_PERCENTAGE_OCCURENCE_THRESHOLD, ids, dt_one_fold_measures, prt=True, file_name="best_single_dt.txt")
			elif alg == "knn":
				feature_selection_before(features, targets, dataset, CV_PERCENTAGE_OCCURENCE_THRESHOLD, ids, knn_one_fold_measures, prt=True, file_name="best_single_knn.txt")
			elif alg == "svm":
				feature_selection_before(features, targets, dataset, CV_PERCENTAGE_OCCURENCE_THRESHOLD, ids, single_svm_fs_one_fold_measures, standardize=True, prt=True, file_name="best_single_svm.txt")	
			else:
				print 'ERROR'	
		
	except IndexError:
		print "Error!! Pass 'all' as argument"
