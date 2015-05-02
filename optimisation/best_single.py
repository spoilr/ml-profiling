import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import *
from parse_theme import *
from feature_selection_before import *
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
		percentage = float(raw_input("Enter percentage."))
		alg = raw_input("Enter algorithm. Choose lr, dt, knn, svm")

		for i in range(100):
			if alg == "lr":
				feature_selection_before(features, targets, dataset, percentage, ids, lr_one_fold_measures_feature_selection, prt=True, file_name="best_single_lr_"+str(percentage)+alg+".txt")
			elif alg == "dt":
				feature_selection_before(features, targets, dataset, percentage, ids, dt_one_fold_measures, prt=True, file_name="best_single_dt_"+str(percentage)+alg+".txt")
			elif alg == "knn":
				feature_selection_before(features, targets, dataset, percentage, ids, knn_one_fold_measures, prt=True, file_name="best_single_knn_"+str(percentage)+alg+".txt")
			elif alg == "svm":
				feature_selection_before(features, targets, dataset, percentage, ids, single_svm_fs_one_fold_measures, standardize=True, prt=True, file_name="best_single_svm_"+str(percentage)+alg+".txt")
			else:
				print 'ERROR'	
		
	except IndexError:
		print "Error!! Pass 'all' as argument"
