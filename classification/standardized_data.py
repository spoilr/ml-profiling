import numpy as np
from split_dataset import *
from parse_theme import *

from sklearn import preprocessing

themes = ['net', 'ill', 'ideo']

class StandardizedData:

	def __init__(self, targets, dataset=None):
		self.dataset = dataset
		self.targets = targets

	def get_known_data_from_theme(self, theme):
		[theme_dataset, theme_features] = parse_theme(theme)
		[known_dataset, known_targets, unk] = split_dataset(theme_dataset, self.targets)
		known_targets = np.asarray(known_targets)
		return [known_dataset, known_targets]

	def thematic_split_and_standardize_dataset(self):
		theme_dataset = []

		net = self.get_known_data_from_theme(themes[0])
		ill = self.get_known_data_from_theme(themes[1])
		ideo = self.get_known_data_from_theme(themes[2])

		net_scaled = preprocessing.scale(net[0])
		ill_scaled = preprocessing.scale(ill[0])
		ideo_scaled = preprocessing.scale(ideo[0])

		theme_dataset.append(net_scaled)
		theme_dataset.append(ill_scaled)
		theme_dataset.append(ideo_scaled)

		# known targets should be all the same for all themes
		assert np.array_equal(net[1], ill[1])
		assert np.array_equal(net[1], ideo[1])
		return theme_dataset, net[1]

	def split_and_standardize_dataset(self):
		assert self.dataset is not None
		[known_dataset, known_targets, unk] = split_dataset(self.dataset, self.targets)
		
		# standardize dataset - Gaussian with zero mean and unit variance
		known_dataset_scaled = preprocessing.scale(known_dataset)
		known_targets = np.asarray(known_targets)
		return known_dataset_scaled, known_targets

	def standardize_dataset(self, data):
		theme_dataset = []

		for i in range(len(data)):
			d_scaled = preprocessing.scale(data[i])
			theme_dataset.append(d_scaled)
			
		return theme_dataset