from sklearn.svm import SVC

def svm_all_vars(dataset, targets):
	# model = SVC(class_weight='auto', C=0.101, gamma=0.001)
	model = SVC(class_weight='auto', C=2.0009999999999999, gamma=0.001)
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_selected_vars(dataset, targets):
	# model = SVC(class_weight='auto', C=0.101000, gamma=0.001000)
	model = SVC(class_weight='auto', C=0.60100000000000009, gamma=0.30100000000000005)
	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_for_features_fusion(dataset, targets):
	# model = SVC(class_weight='auto', C=0.101000, gamma=0.001000)
	# model = SVC(class_weight='auto', C=10.401000, gamma=0.001000)
	# model = SVC(class_weight='auto', C=2.101, gamma=0.001)
	# model = SVC(class_weight='auto', C=2.5009999999999999, gamma=0.001)
	model = SVC(class_weight='auto', C=43.401000000000003, gamma=0.001)

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_subset_features(dataset, targets):
	# model = SVC(class_weight='auto', C=0.101, gamma=0.001)
	# model = SVC(class_weight='auto', C=1.501000, gamma=0.001)
	# model = SVC(class_weight='auto', C=2.101, gamma=0.001)
	model = SVC(class_weight='auto', C=0.60100000000000009, gamma=0.001)

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model