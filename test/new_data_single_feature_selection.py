import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
sys.path.insert(0, 'results/')
from load_data import *
from project_data import *
from parse_theme import *
from split_dataset import *
from selected_features import *
from cv import single_svm_fs_one_fold_measures
from cv import lr_one_fold_measures_feature_selection
from cv import dt_one_fold_measures
from cv import knn_one_fold_measures
from features_from_svm_selection import single_features_90

from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

	training_spreadsheet = Spreadsheet(project_data_file)
	training_data = Data(training_spreadsheet)
	training_targets = training_data.targets

	testing_spreadsheet = Spreadsheet(addendum_data_file, upsampling=False)
	testing_data = Data(testing_spreadsheet, upsampling=False)
	testing_targets = testing_data.targets

	[training_data, features] = parse_theme('all')
	[testing_data, feats] = parse_theme_from_file('all', addendum_data_file)
	assert features == feats

	tech = raw_input("Enter algorithm. Choose between lr, dt, knn, svm")

	[training_data, training_targets, unk] = split_dataset(training_data, training_targets)
	selected_features = single_features_90
	sf = SelectedFeatures(training_data, training_targets, selected_features, features)
	training_data = sf.extract_data_from_selected_features()

	sf = SelectedFeatures(testing_data, testing_targets, selected_features, features)
	testing_data = sf.extract_data_from_selected_features()
		
	# standardize dataset - Gaussian with zero mean and unit variance
	scaler = StandardScaler()

	if tech == 'lr':
		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = lr_one_fold_measures_feature_selection(training_data, testing_data, training_targets, testing_targets)

	elif tech == 'dt':
		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = dt_one_fold_measures(training_data, testing_data, training_targets, testing_targets)
		
	elif tech == 'knn':
		training_data = scaler.fit_transform(training_data)
		testing_data = scaler.transform(testing_data)
		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = knn_one_fold_measures(training_data, testing_data, training_targets, testing_targets)

	elif tech == 'svm':
		training_data = scaler.fit_transform(training_data)
		testing_data = scaler.transform(testing_data)
		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = single_svm_fs_one_fold_measures(training_data, testing_data, training_targets, testing_targets)

	else:
		print 'ERROR technique'	

	print 'Final error %f' % error_rate
	print 'Final accuracy %f' % (1 - error_rate)

	print 'Highval precision %f' % hp
	print 'Highval recall %f' % hr
	print 'Highval f1 %f' % hf
	print 'Civil precision %f' % cp
	print 'Civil recall %f' % cr
	print 'Civil f1 %f' % cf



