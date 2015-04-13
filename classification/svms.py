from sklearn.svm import SVC

def svm_all_vars(dataset, targets):
	
	model = SVC(class_weight='auto', C=2.0009999999999999)	# 0.220000 ; 0.292222

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_selected_vars(dataset, targets):

	model = SVC(class_weight='auto', C=0.700000, gamma=0.300000)	#90 	0.050000 ; 0.314444
	# model = SVC(class_weight='auto', C=1.900000, gamma=0.100000)	#70 	0.040000 ; 0.324444
	# model = SVC(class_weight='auto', C=0.700000, gamma=0.200000)	#50 	0.050000 ; 0.303333

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

# used to train each theme
def svm_for_features_fusion(dataset, targets):

	# no feature selection
	# model = SVC(class_weight='auto', C=2.5009999999999999, gamma=0.01)	# 0.320476 ; 0.354444	0.375000 ; 0.324444		0.162078 ; 0.439167


	model = SVC(class_weight='auto', C=1.000000, gamma=0.200000)	#90		0.000000 ; 0.314444		0.263333 ; 0.247500		0.066667 ; 0.311944
	# model = SVC(class_weight='auto', C=0.700000, gamma=0.200000)	#70 	0.073333 ; 0.370000		0.120000 ; 0.292222		0.291905 ; 0.406111
	# model = SVC(class_weight='auto', C=2.5009999999999999)		#50 	0.123333 ; 0.408889 	0.260000 ; 0.293333		0.258990 ; 0.481944

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

# used for selecting the features
def svm_subset_features(dataset, targets):

	# all
	# model = SVC(class_weight='auto', C=0.7, gamma=0.2)			#90
	# model = SVC(class_weight='auto', C=0.100000, gamma=0.300000)	#70
	# model = SVC(class_weight='auto', C=0.1, gamma=0.1)			#50


	# theme
	model = SVC(class_weight='auto', C=0.100000, gamma=0.100000)	#90
	# model = SVC(class_weight='auto', C=0.100000, gamma=0.100000)	#70
	# model = SVC(class_weight='auto', C=0.700000, gamma=0.100000)	#50

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model