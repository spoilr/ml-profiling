import sys
sys.path.insert(0, 'utils/')
from save_output import save_output
from load_data import *
from labels_fusion import *
from binary_classification_measures import measures
from misclassified_ids import *
from project_data import *
from svms import svm_selected_net
from svms import svm_selected_ill
from svms import svm_selected_ideo

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter

NR_THEMES = 3
themes = ['net', 'ill', 'ideo']
NR_FOLDS = 10

def cv10_ensemble(known_dataset, known_targets, known_dataset_scaled, dt, knn, svm, fusion_algorithm, ids, prt=False, file_name=None):
	error_rates = 0
	hp_rates = 0
	hr_rates = 0
	hf_rates = 0
	cp_rates = 0
	cr_rates = 0
	cf_rates = 0
	for i in range(NR_FOLDS):
		error, hp, hr, hf, cp, cr, cf = cross_validation_ensemble(known_dataset, known_targets, known_dataset_scaled, dt, knn, svm, fusion_algorithm, ids, prt, file_name)
		error_rates += error
		
		hp_rates += hp
		hr_rates += hr
		hf_rates += hf
		cp_rates += cp
		cr_rates += cr
		cf_rates += cf

	if prt and (float(error_rates) / NR_FOLDS) <= 0.4:
		save_output(file_name, error_rates, hp_rates, hr_rates, hf_rates, cp_rates, cr_rates, cf_rates, NR_FOLDS)	

	print 'Final error %f' % (float(error_rates) / NR_FOLDS)
	print 'Final accuracy %f' % (1 - (float(error_rates) / NR_FOLDS))

	print 'Highval precision %f' % (float(hp_rates) / NR_FOLDS)
	print 'Highval recall %f' % (float(hr_rates) / NR_FOLDS)
	print 'Highval f1 %f' % (float(hf_rates) / NR_FOLDS)
	print 'Civil precision %f' % (float(cp_rates) / NR_FOLDS)
	print 'Civil recall %f' % (float(cr_rates) / NR_FOLDS)
	print 'Civil f1 %f' % (float(cf_rates) / NR_FOLDS)	

def cross_validation_ensemble(known_dataset, known_targets, known_dataset_scaled, dt, knn, svm, fusion_algorithm, ids, prt=False, file_name=None):
	kf = StratifiedKFold(known_targets, n_folds=NR_FOLDS, shuffle=True)
	f1_scores = 0
	error_rates = 0
	hp_rates = 0
	hr_rates = 0
	hf_rates = 0
	cp_rates = 0
	cr_rates = 0
	cf_rates = 0
	# cross validation
	for train_index, test_index in kf:
		# print len(test_index)
		# train_index = np.concatenate((train_index, inds), axis=0)
		error, f1, mis_ids, (hp, hr, hf), (cp, cr, cf) = fusion_outputs_ensemble(known_dataset, known_targets, known_dataset_scaled, dt, knn, svm, fusion_algorithm, train_index, test_index, ids)
		
		f1_scores += f1
		error_rates += error
		
		hp_rates += hp
		hr_rates += hr
		hf_rates += hf
		cp_rates += cp
		cr_rates += cr
		cf_rates += cf

	return (float(error_rates) / kf.n_folds), (float(hp_rates) / kf.n_folds), (float(hr_rates) / kf.n_folds), (float(hf_rates) / kf.n_folds), (float(cp_rates) / kf.n_folds), (float(cr_rates) / kf.n_folds), (float(cf_rates) / kf.n_folds)	

def fusion_outputs_ensemble(known_dataset, known_targets, known_dataset_scaled, dt, knn, svm, fusion_algorithm, train_index, test_index, ids):
	misclassified_ids = []
	combined_predictions = []
	y_test = []

	if fusion_algorithm == 'maj':
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids, dt, ind=False)
		combined_predictions_dt = majority_vote(predictions, y_test, accuracies)
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset_scaled, known_targets, train_index, test_index, ids, knn, ind=False)
		combined_predictions_knn = majority_vote(predictions, y_test, accuracies)
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset_scaled, known_targets, train_index, test_index, ids, svm, ind=True)
		combined_predictions_svm = majority_vote(predictions, y_test, accuracies)

		combined_predictions = []
		assert len(combined_predictions_dt) == len(combined_predictions_knn)
		assert len(combined_predictions_dt) == len(combined_predictions_svm)

		for i in range(len(combined_predictions_dt)):
			data = Counter([combined_predictions_dt[i], combined_predictions_knn[i], combined_predictions_svm[i]])
			combined_predictions.append(data.most_common(1)[0][0])

	elif fusion_algorithm == 'wmaj':
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids, dt, ind=False)
		combined_predictions_dt, weights = weighted_majority(predictions, y_test)
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset_scaled, known_targets, train_index, test_index, ids, knn, ind=False)
		combined_predictions_knn, weights = weighted_majority(predictions, y_test)
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset_scaled, known_targets, train_index, test_index, ids, svm, ind=True)
		combined_predictions_svm, weights = weighted_majority(predictions, y_test)

		combined_predictions = []
		assert len(combined_predictions_dt) == len(combined_predictions_knn)
		assert len(combined_predictions_dt) == len(combined_predictions_svm)

		for i in range(len(combined_predictions_dt)):
			data = Counter([combined_predictions_dt[i], combined_predictions_knn[i], combined_predictions_svm[i]])
			combined_predictions.append(data.most_common(1)[0][0])

	elif fusion_algorithm == 'svm':
		y_test, predictions, combined_predictions_dt, misclassified_ids = svm_fusion(known_dataset, known_targets, train_index, test_index, ids, dt, ind=False)
		y_test, predictions, combined_predictions_knn, misclassified_ids = svm_fusion(known_dataset_scaled, known_targets, train_index, test_index, ids, knn, ind=False)
		y_test, predictions, combined_predictions_svm, misclassified_ids = svm_fusion(known_dataset_scaled, known_targets, train_index, test_index, ids, svm, ind=True)

		combined_predictions = []
		assert len(combined_predictions_dt) == len(combined_predictions_knn)
		assert len(combined_predictions_dt) == len(combined_predictions_svm)

		for i in range(len(combined_predictions_dt)):
			data = Counter([combined_predictions_dt[i], combined_predictions_knn[i], combined_predictions_svm[i]])
			combined_predictions.append(data.most_common(1)[0][0])

	elif fusion_algorithm == 'nn':
		print 'not done'
	else:
		print 'Error parsing algorithm'

	# print '###############'
	# print 'Y_PRED %s' % str(predictions)
	# print 'Y_TEST %s' % str(y_test)
	# print 'COMBINED %s' % str(combined_predictions)
	# print '###############'

	(hp, hr, hf), (cp, cr, cf) = measures(y_test, combined_predictions)

	error = (float(sum((combined_predictions - y_test)**2)) / len(y_test))
	f1 = f1_score(combined_predictions, y_test)
	return error, f1, misclassified_ids, (hp, hr, hf), (cp, cr, cf)

def cv10(known_dataset, known_targets, fusion_algorithm, ids, algorithm, prt=False, file_name=None, ind=False):
	error_rates = 0
	hp_rates = 0
	hr_rates = 0
	hf_rates = 0
	cp_rates = 0
	cr_rates = 0
	cf_rates = 0
	for i in range(NR_FOLDS):
		error, hp, hr, hf, cp, cr, cf = cross_validation(known_dataset, known_targets, fusion_algorithm, ids, algorithm, prt, file_name, ind)		
		error_rates += error
		
		hp_rates += hp
		hr_rates += hr
		hf_rates += hf
		cp_rates += cp
		cr_rates += cr
		cf_rates += cf

	if prt and (float(error_rates) / NR_FOLDS) <= 0.5:
		save_output(file_name, error_rates, hp_rates, hr_rates, hf_rates, cp_rates, cr_rates, cf_rates, NR_FOLDS)	

	print 'Final error %f' % (float(error_rates) / NR_FOLDS)
	print 'Final accuracy %f' % (1 - (float(error_rates) / NR_FOLDS))

	print 'Highval precision %f' % (float(hp_rates) / NR_FOLDS)
	print 'Highval recall %f' % (float(hr_rates) / NR_FOLDS)
	print 'Highval f1 %f' % (float(hf_rates) / NR_FOLDS)
	print 'Civil precision %f' % (float(cp_rates) / NR_FOLDS)
	print 'Civil recall %f' % (float(cr_rates) / NR_FOLDS)
	print 'Civil f1 %f' % (float(cf_rates) / NR_FOLDS)	


def cross_validation(known_dataset, known_targets, fusion_algorithm, ids, algorithm, prt=False, file_name=None, ind=False):
	misclassified_ids = []

	kf = StratifiedKFold(known_targets, n_folds=NR_FOLDS, shuffle=True)
	f1_scores = 0
	error_rates = 0
	hp_rates = 0
	hr_rates = 0
	hf_rates = 0
	cp_rates = 0
	cr_rates = 0
	cf_rates = 0
	# cross validation
	for train_index, test_index in kf:
		# print len(test_index)
		# train_index = np.concatenate((train_index, inds), axis=0)
		error, f1, mis_ids, (hp, hr, hf), (cp, cr, cf) = fusion_outputs(known_dataset, known_targets, train_index, test_index, fusion_algorithm, ids, algorithm, ind)
		
		f1_scores += f1
		error_rates += error
		
		hp_rates += hp
		hr_rates += hr
		hf_rates += hf
		cp_rates += cp
		cr_rates += cr
		cf_rates += cf
		misclassified_ids += mis_ids


	misclassified_ids = set(misclassified_ids)	
	# print '########## MISCLASSIFIED ########## %d \n %s' % (len(misclassified_ids), str(misclassified_ids))
	# print 'Final f1 %f' % (float(f1_scores) / kf.n_folds)
	# print 'Final error %f' % (float(error_rates) / kf.n_folds)
	# print 'Final accuracy %f' % (1 - (float(error_rates) / kf.n_folds))

	# print 'Highval precision %f' % (float(hp_rates) / kf.n_folds)
	# print 'Highval recall %f' % (float(hr_rates) / kf.n_folds)
	# print 'Highval f1 %f' % (float(hf_rates) / kf.n_folds)
	# print 'Civil precision %f' % (float(cp_rates) / kf.n_folds)
	# print 'Civil recall %f' % (float(cr_rates) / kf.n_folds)
	# print 'Civil f1 %f' % (float(cf_rates) / kf.n_folds)

	return (float(error_rates) / kf.n_folds), (float(hp_rates) / kf.n_folds), (float(hr_rates) / kf.n_folds), (float(hf_rates) / kf.n_folds), (float(cp_rates) / kf.n_folds), (float(cr_rates) / kf.n_folds), (float(cf_rates) / kf.n_folds)	


def fusion_outputs(known_dataset, known_targets, train_index, test_index, fusion_algorithm, ids, algorithm, ind):
	misclassified_ids = []
	combined_predictions = []
	y_test = []

	if fusion_algorithm == 'maj':
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids, algorithm, ind)
		combined_predictions = majority_vote(predictions, y_test, accuracies)

	elif fusion_algorithm == 'wmaj':
		predictions, y_test, accuracies, misclassified_ids = combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids, algorithm, ind)
		combined_predictions, weights = weighted_majority(predictions, y_test)

	elif fusion_algorithm == 'svm':
		y_test, predictions, combined_predictions, misclassified_ids = svm_fusion(known_dataset, known_targets, train_index, test_index, ids, algorithm, ind)

	elif fusion_algorithm == 'nn':
		print 'not done'
	else:
		print 'Error parsing algorithm'

	# print '###############'
	# print 'Y_PRED %s' % str(predictions)
	# print 'Y_TEST %s' % str(y_test)
	# print 'COMBINED %s' % str(combined_predictions)
	# print '###############'

	(hp, hr, hf), (cp, cr, cf) = measures(y_test, combined_predictions)

	error = (float(sum((combined_predictions - y_test)**2)) / len(y_test))
	f1 = f1_score(combined_predictions, y_test)
	return error, f1, misclassified_ids, (hp, hr, hf), (cp, cr, cf)


# Training and testing sets initially
# 2/3 are used to train the algorithm and 1/3 is used to train(after the output is obtained) the fusion SVM
def svm_fusion(known_dataset, known_targets, train_index, test_index, ids, algorithm, ind):
	misclassified_ids = []

	training_predictions = []
	predictions = []
	fusion_Y_train = []
	y_train, final_y_test = known_targets[train_index], known_targets[test_index]

	kf = StratifiedKFold(y_train, n_folds=3)
	curr = 0
	for inner_train_index, inner_test_index in kf:

		for i in range(0, NR_THEMES):
			X_train, final_X_test = known_dataset[i][train_index], known_dataset[i][test_index]
			svm_X_train, svm_Y_train = X_train[inner_train_index], y_train[inner_train_index]
			fusion_X_train, fusion_Y_train = X_train[inner_test_index], y_train[inner_test_index]


			if ind:
				if i == 0:
					model = svm_selected_net(svm_X_train, svm_Y_train)
				elif i == 1:
					model = svm_selected_ill(svm_X_train, svm_Y_train)
				elif i == 2:
					model = svm_selected_ideo(svm_X_train, svm_Y_train)
			else:
				model = algorithm(svm_X_train, svm_Y_train)

			training_predictions.append(model.predict(fusion_X_train))
			predictions.append(model.predict(final_X_test))
			misclassified_ids += add_misclassified_ids(model, test_index, known_dataset[i], known_targets, ids)

		curr+=1
		if curr == 1:
			break

	training_pred_input = np.vstack(training_predictions).T
	fusion_model = inner_svm(training_pred_input, fusion_Y_train)

	pred_input = np.vstack(predictions).T
	combined_predictions = fusion_model.predict(pred_input)

	return final_y_test, predictions, combined_predictions.tolist(), misclassified_ids		


def inner_svm(dataset, targets):
	# model = SVC(class_weight='auto', C=0.5, gamma=0.10000000000000001)
	model = SVC(class_weight='auto', C=16, gamma=16)
	model.fit(dataset, targets)
	return model

def dt(dataset, targets):
	model = DecisionTreeClassifier(criterion='entropy')
	model.fit(dataset, targets)
	return model

def lr(dataset, targets):
	model = LogisticRegression(class_weight='auto', C=0.06)
	model.fit(dataset, targets)
	return model

def lr_feature_selection(dataset, targets):
	model = LogisticRegression(class_weight='auto', C=20.960000)
	model.fit(dataset, targets)
	return model

def knn(dataset, targets):
	model = KNeighborsClassifier(weights='distance', n_neighbors=5)
	model.fit(dataset, targets)
	return model		

# called majority because it is used in both cases of majority_voting and weighted_majority voting.
def combine_predictions_one_fold_using_majority(known_dataset, known_targets, train_index, test_index, ids, algorithm, ind):
	misclassified_ids = []

	predictions = []
	accuracies = []
	y_train, y_test = known_targets[train_index], known_targets[test_index]
	for i in range(0, NR_THEMES):
		X_train, X_test = known_dataset[i][train_index], known_dataset[i][test_index]

		if ind:
			if i == 0:
				model = svm_selected_net(X_train, y_train)
			elif i == 1:
				model = svm_selected_ill(X_train, y_train)
			elif i == 2:
				model = svm_selected_ideo(X_train, y_train)
		else:
			model = algorithm(X_train, y_train)

		accuracy = model.score(X_test, y_test)
		# print 'Model score for %s is %f' % (themes[i], accuracy)
		y_pred = model.predict(X_test)
		predictions.append(y_pred)
		accuracies.append(accuracy)
		misclassified_ids += add_misclassified_ids(model, test_index, known_dataset[i], known_targets, ids)
	
	predictions = np.array((predictions[0], predictions[1], predictions[2]), dtype=float)
	print "%s \n vs %s \n\n" % (str(predictions), str(y_test))
	return predictions, y_test, accuracies, misclassified_ids

