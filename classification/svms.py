from sklearn.svm import SVC

def svm_all_vars(dataset, targets):
	model = SVC(class_weight='auto')
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_selected_vars(dataset, targets):
	model = SVC(class_weight='auto')
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_for_features_fusion(dataset, targets):
	model = SVC(class_weight='auto')
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_subset_features(dataset, targets):
	model = SVC(class_weight='auto')
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model