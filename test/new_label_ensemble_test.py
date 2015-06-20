import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
sys.path.insert(0, 'optimisation/')
sys.path.insert(0, 'results/')
sys.path.insert(0, 'feature context/')
from load_data_d import *
from load_data_v import *
from standardized_data import *
from feature_selection_before import *
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from cv import ensemble_one_fold_measures
from cv import dt
from cv import knn
from fusion import dt
from fusion import knn
from fusion import inner_svm
from svms import svm_all_vars
from svms import svm_selected_vars
from svms import svm_selected_net
from svms import svm_selected_ill
from svms import svm_selected_ideo
from svms import svm_selected_for_features_fusion
from svms import svm_for_features_fusion
from ensemble_theme_test import svm_vote
from ensemble_theme_test import fusion
from ensemble_theme_test import weighted_majority_theme
from features_from_svm_selection import single_features_90
from replace_missing_values import *
from opt_fusion_svm import combine_and_process_dataset
from label_binary_classification_measures import measures
from labels_fusion import *
from weights import *
from save_output import *
from ensemble_single_test import ensemble_single
from ensemble_single_feature_selection_test import ensemble_single_feature_selection
from ensemble_theme_test import ensemble_theme
from ensemble_theme_feature_selection_test import ensemble_theme_feature_selection

import numpy as np
from sklearn.preprocessing import StandardScaler


net_90 = set(['LettersPost', 'Education', 'Discriminate', 'LiveAlone', 'BombManuals', 'InteractNet', 'OtherKnowledge', 'FurtherAttacks', 'NotCareInjustice', 'Getaway', 'Involve', 'NewMedia', 'Stockpile', 'DryRuns', 'RecruitNetGroup', 'Interrupt', 'Financial', 'TargetTyp', 'WarningLettersStatements'])
ill_90 = set(['MentalIll', 'Stress', 'Education', 'Financial', 'LiveAlone', 'BombManuals', 'LettersPost', 'FurtherAttacks', 'NotCareInjustice', 'Isolated', 'Getaway', 'DryRuns', 'HurtOthers', 'NewMedia', 'Stockpile', 'OtherKnowledge', 'Interrupt', 'TargetTyp', 'WarningLettersStatements', 'AwareGriev'])
ideo_90 = set(['Discriminate', 'NotCareInjustice', 'FurtherAttacks', 'Religion', 'AwareIdeo', 'RelatStat', 'Denounce', 'IdeoChangeInt', 'Interrupt', 'LettersPost', 'BombManuals', 'Contradict', 'Legitimise', 'Getaway', 'Ideology', 'Stockpile', 'RecruitNetGroup', 'Propaganda', 'Financial', 'LiveAlone', 'LocPubPriv', 'OtherKnowledge', 'ReligChangeInt', 'TargetTyp', 'Education'])
single_features_90 = set(['MilExp', 'Discriminate', 'InteractNet', 'NotCareInjustice', 'FurtherAttacks', 'Isolated', 'Religion', 'OtherInv', 'AwareIdeo', 'MultiAttackMeth', 'Insanity', 'RelatStat', 'Denounce', 'IdeoChangeInt', 'Interrupt', 'WarningLettersStatements', 'MentalIll', 'Stress', 'BombManuals', 'Contradict', 'Legitimise', 'Getaway', 'ReligChangeInt', 'Ideology', 'HurtOthers', 'Stockpile', 'RecruitNetGroup', 'LettersPost', 'Implement', 'Propaganda', 'NewMedia', 'Financial', 'LocPubPriv', 'LiveAlone', 'OtherKnowledge', 'Involve', 'DryRuns', 'TargetTyp', 'AwareGriev', 'Education'])

themes = ['net', 'ill', 'ideo']

class ThematicDataCombined:

	def __init__(self, targets, data, dataset=None):
		self.dataset = dataset
		self.targets = targets
		self.data = data

	def get_known_data_from_theme(self, theme):
		[theme_dataset, theme_features] = parse_categ(theme, self.data)
		[known_dataset, known_targets, unk] = split_dataset(theme_dataset, self.targets)
		known_targets = np.asarray(known_targets)
		return [known_dataset, known_targets]

	def thematic_split(self):
		theme_dataset = []

		net = self.get_known_data_from_theme(themes[0])
		ill = self.get_known_data_from_theme(themes[1])
		ideo = self.get_known_data_from_theme(themes[2])

		theme_dataset.append(np.array(net[0]))
		theme_dataset.append(np.array(ill[0]))
		theme_dataset.append(np.array(ideo[0]))

		# known targets should be all the same for all themes
		assert np.array_equal(net[1], ill[1])
		assert np.array_equal(net[1], ideo[1])
		return theme_dataset, net[1]


def thematic_data_from_feature_selection(orig_targets, theme, percentage, data):
	[dataset, features] = parse_categ(theme, data)
	[known_dataset, known_targets, unk] = split_dataset(dataset, orig_targets)
	
	known_targets = np.asarray(known_targets)
	selected_features = select_features(percentage, theme)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)

	print '####### %s FEATURES ####### %d %s' % (theme, len(selected_features), str(selected_features)) 

	return sf.extract_data_from_selected_features(), known_targets

def select_features(percentage, theme):
	if theme == 'net' and percentage == 0.9:
		return net_90
	elif theme == 'ill' and percentage == 0.9:
		return ill_90
	elif theme == 'ideo' and percentage == 0.9:
		return ideo_90
	else:
		print 'ERROR in percentage - theme'

def combine_data_from_feature_selection(orig_targets, percentage, data_info):
	combined_dataset = []
	targets = []
	for theme in themes:
		data, targets = thematic_data_from_feature_selection(orig_targets, theme, percentage, data_info)
		combined_dataset.append(data)
	return combined_dataset, targets	

def combine_data_from_feature_selection_from_file(orig_targets, percentage, file_name, data_info):
	combined_dataset = []
	targets = []
	for theme in themes:
		data, targets = thematic_data_from_feature_selection(orig_targets, theme, percentage, data_info)
		combined_dataset.append(data)
	return combined_dataset, targets

def parse_categ(theme, data):
	if theme == 'ill':
		dataset = data.extract_illness_examples()
		features = data.illness_features
	elif theme == 'net':
		dataset = data.extract_network_examples()
		features = data.network_features
	elif theme == 'ideo':
		dataset = data.extract_ideology_examples()
		features = data.ideology_features	
	elif theme == 'all':
		dataset = data.extract_selected_examples()
		features = data.selected_features
	else:
		print 'Error parsing theme'

	return [dataset, features]

def split_dataset(dataset, targets):
	unknowns = []
	known_dataset = []
	known_targets = []
	for i in range(0, len(targets)):
		known_dataset.append(dataset[i])
		known_targets.append(targets[i])

	return [np.array(known_dataset), known_targets, np.array(unknowns)]

if __name__ == "__main__":

	label = raw_input("Label: d, v.\n")
	method = raw_input("Method: s, sf, t, tf.\n")

	if label == "d":
		training_spreadsheet = SpreadsheetD('/Users/oana/Downloads/ip/project data no d.xlsx')
		training_data = DataD(training_spreadsheet)
		training_targets = training_data.targets

		testing_spreadsheet = Spreadsheet('/Users/oana/Downloads/ip/addendum d.xlsx', upsampling=False)
		testing_data = DataD(testing_spreadsheet, upsampling=False)
		testing_targets = testing_data.targets
	elif label == "v":
		training_spreadsheet = SpreadsheetV('/Users/oana/Downloads/ip/project data no v.xlsx')
		training_data = DataV(training_spreadsheet)
		training_targets = training_data.targets

		testing_spreadsheet = Spreadsheet('/Users/oana/Downloads/ip/addendum v.xlsx', upsampling=False)
		testing_data = DataV(testing_spreadsheet, upsampling=False)
		testing_targets = testing_data.targets
	else:
		print'ERROR label'	

	if "t" in method:
		fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
		net_scaler = StandardScaler()
		ill_scaler = StandardScaler()
		ideo_scaler = StandardScaler()

	if "s" in method:
		[training_data, features] = parse_categ('all', training_data)
		[testing_data, feats] = parse_categ('all', testing_data)
		assert features == feats
			

	if method == "s":
		[training_data, training_targets, unk] = split_dataset(training_data, training_targets)
	elif method == "sf":
		[training_data, training_targets, unk] = split_dataset(training_data, training_targets)
		selected_features = single_features_90
		sf = SelectedFeatures(training_data, training_targets, selected_features, features)
		training_data = sf.extract_data_from_selected_features()

		sf = SelectedFeatures(testing_data, testing_targets, selected_features, features)
		testing_data = sf.extract_data_from_selected_features()
	elif method == "t":
		tdc = ThematicDataCombined(training_targets, training_data)
		training_data, training_targets = tdc.thematic_split() 	

		tdc = ThematicDataCombined(testing_targets, testing_data)
		testing_data, testing_targets = tdc.thematic_split()
	elif method == "tf":
		training_data, training_targets = combine_data_from_feature_selection(training_targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD, training_data)
		testing_data, testing_targets = combine_data_from_feature_selection_from_file(testing_targets, CV_PERCENTAGE_OCCURENCE_THRESHOLD, addendum_data_file, testing_data)
	else:
		print 'ERROR method'	


	if "s" in method:
		# standardize dataset - Gaussian with zero mean and unit variance
		scaler = StandardScaler()

		testing_data = replace_missings(testing_data)
		training_data_scaled = scaler.fit_transform(training_data)
		testing_data_scaled = scaler.transform(testing_data)

	if "t" in method:
		testing_data = replace_missings_thematic(testing_data)

		training_data_scaled = []
		training_data_scaled.append(net_scaler.fit_transform(training_data[0]))
		training_data_scaled.append(ill_scaler.fit_transform(training_data[1]))
		training_data_scaled.append(ideo_scaler.fit_transform(training_data[2]))

		testing_data_scaled = []
		testing_data_scaled.append(net_scaler.transform(testing_data[0]))
		testing_data_scaled.append(ill_scaler.transform(testing_data[1]))
		testing_data_scaled.append(ideo_scaler.transform(testing_data[2]))


	if method == "s":
		file_name = "ensemble_single.txt"
		error_rate, f1, (hp, hr, hf), (cp, cr, cf) = ensemble_single(training_data, testing_data, training_data_scaled, testing_data_scaled, training_targets, testing_targets, dt, knn, svm_all_vars)
	elif method == "sf":
		file_name = "ensemble_single_feature_selection.txt"
		error_rate, f1, (hp, hr, hf), (cp, cr, cf) = ensemble_single_feature_selection(training_data, testing_data, training_data_scaled, testing_data_scaled, training_targets, testing_targets, dt, knn, svm_selected_vars)
	elif method == "t":
		file_name = "ensemble_theme.txt"
		error_rate, (hp, hr, hf), (cp, cr, cf) = ensemble_theme(training_data, training_data_scaled, training_targets, testing_data, testing_data_scaled, testing_targets, fusion_algorithm)
	elif method == "tf":
		file_name = "ensemble_theme_feature_selection.txt"
		error_rate, (hp, hr, hf), (cp, cr, cf) = ensemble_theme_feature_selection(training_data, training_data_scaled, training_targets, testing_data, testing_data_scaled, testing_targets, fusion_algorithm)
	else:
		print 'ERROR method output'	

