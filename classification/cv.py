import sys
sys.path.insert(0, 'utils/')
from save_output import save_output
from binary_classification_measures import measures
from misclassified_ids import *
from svms import svm_selected_vars
from svms import svm_all_vars

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score

NR_FOLDS = 10

def cv10(known_dataset, known_targets, ids, one_fold_measures, prt=False, file_name=None):
	error_rates = 0
	hp_rates = 0
	hr_rates = 0
	hf_rates = 0
	cp_rates = 0
	cr_rates = 0
	cf_rates = 0
	for i in range(NR_FOLDS):
		error, hp, hr, hf, cp, cr, cf = cross_validation(known_dataset, known_targets, ids, one_fold_measures, prt, file_name)		
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

def cross_validation(known_dataset, known_targets, ids, one_fold_measures, prt=False, file_name=None):
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
	for train_index, test_index in kf:
		X_train, X_test = known_dataset[train_index], known_dataset[test_index]
		y_train, y_test = known_targets[train_index], known_targets[test_index]
		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = one_fold_measures(X_train, X_test, y_train, y_test)
		f1_scores += f1
		error_rates += error_rate

		hp_rates += hp
		hr_rates += hr
		hf_rates += hf
		cp_rates += cp
		cr_rates += cr
		cf_rates += cf
		misclassified_ids += add_misclassified_ids(model, test_index, known_dataset, known_targets, ids)

	# print '########## MISCLASSIFIED ########## %d %s' % (len(misclassified_ids), str(misclassified_ids))
	
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

def single_svm_one_fold_measures(X_train, X_test, y_train, y_test):
	model = svm_all_vars(X_train, y_train)
	# print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	f1 = f1_score(y_test, y_pred)
	(hp, hr, hf), (cp, cr, cf) = measures(y_test, y_pred)

	# print_pred_test(y_pred, y_test)
	return error_rate, f1, model, (hp, hr, hf), (cp, cr, cf)

def single_svm_fs_one_fold_measures(X_train, X_test, y_train, y_test):
	model = svm_selected_vars(X_train, y_train)
	# print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	f1 = f1_score(y_test, y_pred)
	(hp, hr, hf), (cp, cr, cf) = measures(y_test, y_pred)

	# print_pred_test(y_pred, y_test)
	return error_rate, f1, model, (hp, hr, hf), (cp, cr, cf)

def dt_one_fold_measures(X_train, X_test, y_train, y_test):
	model = dt(X_train, y_train)
	# print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	f1 = f1_score(y_test, y_pred)
	(hp, hr, hf), (cp, cr, cf) = measures(y_test, y_pred)

	# print_pred_test(y_pred, y_test)
	return error_rate, f1, model, (hp, hr, hf), (cp, cr, cf)	

def dt(dataset, targets):
	model = DecisionTreeClassifier(criterion='entropy')
	model.fit(dataset, targets)
	return model

def knn_one_fold_measures(X_train, X_test, y_train, y_test):
	model = knn(X_train, y_train)
	# print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	f1 = f1_score(y_test, y_pred)
	(hp, hr, hf), (cp, cr, cf) = measures(y_test, y_pred)

	# print_pred_test(y_pred, y_test)
	return error_rate, f1, model, (hp, hr, hf), (cp, cr, cf)	

def knn(dataset, targets):
	model = KNeighborsClassifier(weights='distance', n_neighbors=5)
	model.fit(dataset, targets)
	return model

def lr_one_fold_measures(X_train, X_test, y_train, y_test):
	model = lr(X_train, y_train)
	# print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	f1 = f1_score(y_test, y_pred)
	(hp, hr, hf), (cp, cr, cf) = measures(y_test, y_pred)

	# print_pred_test(y_pred, y_test)
	return error_rate, f1, model, (hp, hr, hf), (cp, cr, cf)	

def lr_one_fold_measures_feature_selection(X_train, X_test, y_train, y_test):
	model = lr_feature_selection(X_train, y_train)
	# print 'Model score %f' % model.score(X_test, y_test)
	y_pred = model.predict(X_test)
	error_rate = (float(sum((y_pred - y_test)**2)) / len(y_test))
	f1 = f1_score(y_test, y_pred)
	(hp, hr, hf), (cp, cr, cf) = measures(y_test, y_pred)

	# print_pred_test(y_pred, y_test)
	return error_rate, f1, model, (hp, hr, hf), (cp, cr, cf)

def lr(dataset, targets):
	model = LogisticRegression(class_weight='auto', C=0.060000000000000005)
	model.fit(dataset, targets)
	return model		

def lr_feature_selection(dataset, targets):
	model = LogisticRegression(class_weight='auto', C=48.359999999999999)
	model.fit(dataset, targets)
	return model	


def print_pred_test(pred, test):
	print 'PRED ' + str(pred)
	print 'TEST ' + str(test)
