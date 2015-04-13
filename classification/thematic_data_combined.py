'''
Combine thematic data.
'''

import sys
sys.path.insert(0, 'feature context/')
import numpy as np
from split_dataset import *
from parse_theme import *
from feature_selection_cv import *
from selected_features import *

themes = ['net', 'ill', 'ideo']

class ThematicDataCombined:

	def __init__(self, targets, dataset=None):
		self.dataset = dataset
		self.targets = targets

	def get_known_data_from_theme(self, theme):
		[theme_dataset, theme_features] = parse_theme(theme)
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


def thematic_data_from_feature_selection(orig_targets, theme, percentage):
	[dataset, features] = parse_theme(theme)
	[known_dataset, known_targets, unk] = split_dataset(dataset, orig_targets)
	
	known_targets = np.asarray(known_targets)

	cv_features = features_cross_validation(known_dataset, known_targets, features)
	selected_features = select_final_features_from_cv(cv_features, percentage)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)

	print '####### %s FEATURES ####### %d %s' % (theme, len(selected_features), str(selected_features)) 

	return sf.extract_data_from_selected_features(), known_targets

def combine_data_from_feature_selection(orig_targets, percentage):
	combined_dataset = []
	targets = []
	for theme in themes:
		data, targets = thematic_data_from_feature_selection(orig_targets, theme, percentage)
		combined_dataset.append(data)
	return combined_dataset, targets	
