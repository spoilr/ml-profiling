import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
from load_data import *
from labels_fusion import weighted_majority
from project_data import *
from standardized_data import *
from thematic_data_combined import combine_data_from_feature_selection

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from collections import Counter
import itertools

NR_THEMES = 3
themes = ['net', 'ill', 'ideo']

class BestSVM:
	def __init__(self, c_net, g_net, c_ill, g_ill, c_ideo, g_ideo):
		self.c_net = c_net
		self.g_net = g_net
		self.c_ill = c_ill
		self.g_ill = g_ill
		self.c_ideo = c_ideo
		self.g_ideo = g_ideo

	def svm_net(self, dataset, targets):
		model = SVC(class_weight='auto', C=self.c_net, gamma=self.g_net)
		model.fit(dataset, targets)
		return model

	def svm_ill(self, dataset, targets):
		model = SVC(class_weight='auto', C=self.c_ill, gamma=self.g_ill)
		model.fit(dataset, targets)
		return model

	def svm_ideo(self, dataset, targets):
		model = SVC(class_weight='auto', C=self.c_ideo, gamma=self.g_ideo)
		model.fit(dataset, targets)
		return model		

	def to_string(self):
		return 'c_net %f, g_net %f ||| c_ill %f, g_ill %f ||| c_ideo %f, g_ideo %f' % (self.c_net, self.g_net, self.c_ill, self.g_ill, self.c_ideo, self.g_ideo)

def cross_validation(known_dataset, known_targets, best_svm):
	kf = StratifiedKFold(known_targets, n_folds=10)
	f1_scores = 0
	error_rates = 0
	# cross validation
	for train_index, test_index in kf:
		error, f1 = fusion_outputs(known_dataset, known_targets, train_index, test_index, best_svm)
		
		f1_scores += f1
		error_rates += error

	return float(error_rates) / kf.n_folds, float(f1_scores) / kf.n_folds

def fusion_outputs(known_dataset, known_targets, train_index, test_index, best_svm):
	predictions, y_test = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, best_svm)
	combined_predictions = weighted_majority(predictions, y_test)
	error = (float(sum((combined_predictions - y_test)**2)) / len(y_test))
	f1 = f1_score(combined_predictions, y_test)
	return error, f1

def combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, best_svm):
	predictions = []
	y_train, y_test = known_targets[train_index], known_targets[test_index]
	for i in range(0, NR_THEMES):
		X_train, X_test = known_dataset[i][train_index], known_dataset[i][test_index]

		if i == 0:
			model = best_svm.svm_net(X_train, y_train)
		elif i == 1:
			model = best_svm.svm_ill(X_train, y_train)
		elif i == 2:
			model = best_svm.svm_ideo(X_train, y_train)

		y_pred = model.predict(X_test)
		predictions.append(y_pred)
	
	predictions = np.array((predictions[0], predictions[1], predictions[2]), dtype=float)
	return predictions, y_test

def params():
	begin = 0.1
	end = 2.5
	C_range = np.arange(begin, end, 0.1)
	gamma_range = np.arange(begin, 1.3, 0.1)
	return C_range, gamma_range



if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	C_net, g_net = params()	
	C_ill, g_ill = params()	
	C_ideo, g_ideo = params()	
	
	C_range = [C_net, C_ill, C_ideo]
	g_range = [g_net, g_ill, g_ideo]

	for cs in itertools.product(*C_range):
		for gs in itertools.product(*g_range):
			c_net = cs[0]
			c_ill = cs[1]
			c_ideo = cs[2]
			g_net = gs[0]
			g_ill = gs[1]
			g_ideo = gs[2]
			best_svm = BestSVM(c_net, g_net, c_ill, g_ill, c_ideo, g_ideo)

			combined_dataset, targets = combine_data_from_feature_selection(targets, 0.9)

			std = StandardizedData(targets)
			dataset = std.standardize_dataset(combined_dataset)  

			f1, error = cross_validation(dataset, targets, best_svm)
			
			if error <= 0.33 and f1 > 0:
				with open("result.txt", "a") as myfile:	
					myfile.write('\n##############################\n')
				with open("result.txt", "a") as myfile:
					myfile.write(best_svm.to_string())
				with open("result.txt", "a") as myfile:	
					myfile.write('\nerror_maj %f' % error)
				with open("result.txt", "a") as myfile:	
					myfile.write('\nf1 %f' % f1)	
	    	
			print best_svm.to_string()

	
