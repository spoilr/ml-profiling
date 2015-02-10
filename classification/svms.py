from sklearn.svm import SVC

def svm_all_vars(dataset, targets):
	# model = SVC(class_weight='auto', C=0.90000999999999987, gamma=0.30001)
	model = SVC(class_weight='auto', C=0.9, gamma=0.303333)
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_selected_vars(dataset, targets):
	# model = SVC(class_weight='auto', C=0.90000999999999987, gamma=12.600009999999999)
	model = SVC(class_weight='auto', C=0.9, gamma=14.700010)
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_for_features_fusion(dataset, targets):
	# model = SVC(class_weight='auto', C=0.90000999999999987, gamma=0.30001)
	model = SVC(class_weight='auto', C=0.9, gamma=14.700010)
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_subset_features(dataset, targets):
	# model = SVC(class_weight='auto', C=0.90000999999999987, gamma=0.30001)
	model = SVC(class_weight='auto', C=0.9, gamma=14.700010)
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model