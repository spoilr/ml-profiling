import numpy as np

class SelectedFeatures:

	def __init__(self, known_dataset, known_targets, selected_features, features):
		self.known_dataset = known_dataset
		self.known_targets = known_targets
		self.selected_features = selected_features
		self.features = features

	def extract_data_from_selected_features(self):
		indices = self.extract_indices_of_selected_features()
		dataset_of_selected_features = self.extract_examples_of_selected_features(indices)
		return dataset_of_selected_features

	def extract_indices_of_selected_features(self):
		indices = []

		for x in self.selected_features:
			try:
				indices.append(self.features.index(x))
			except ValueError:
				print "Theme %s is not in the Features!" % x	

		return indices

	def extract_examples_of_selected_features(self, indices):
		num_rows = len(self.known_targets)
		examples = []

		for curr_row in range(num_rows):
			row = []
			for ind in indices:
				row.append(self.known_dataset[curr_row, ind])
			examples.append(row)	

		assert len(examples) == len(self.known_targets)
		return np.asarray(examples)	