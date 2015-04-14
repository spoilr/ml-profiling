from sklearn.svm import SVC

class BestSVM:
	def __init__(self, c_subset, g_subset, c_fusion, g_fusion):
		self.c_subset = c_subset
		self.g_subset = g_subset
		self.c_fusion = c_fusion
		self.g_fusion = g_fusion

	# svm for fusion of outputs of the themes	
	def inner_svm(self, dataset, targets):
		model = SVC(class_weight='auto')
		model.fit(dataset, targets)
		return model

	# used for selecting the features	
	def svm_subset_features(self, dataset, targets):
		model = SVC(class_weight='auto', C=self.c_subset, gamma=self.g_subset)
		model.fit(dataset, targets)
		return model

	# used to train each theme	
	def svm_for_features_fusion(self, dataset, targets):
		model = SVC(class_weight='auto', C=self.c_fusion, gamma=self.g_fusion)
		model.fit(dataset, targets)
		return model

	def to_string(self):
		return 'c_subset %f, g_subset %f ||| c_fusion %f, g_fusion %f' % (self.c_subset, self.g_subset, self.c_fusion, self.g_fusion)


class BestFeatureSVM:
	def __init__(self, c, g):
		self.c = c
		self.g = g

	def svm_subset_features(self, dataset, targets):
		model = SVC(class_weight='auto', C=self.c, gamma=self.g)
		model.fit(dataset, targets)
		return model

	def to_string(self):
		return 'c %f, g %f' % (self.c, self.g)	