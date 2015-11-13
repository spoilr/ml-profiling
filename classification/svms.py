from sklearn.svm import SVC

def svm_all_vars(dataset, targets):
	
	model = SVC(class_weight='auto', C=2.0009999999999999)

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_selected_vars(dataset, targets):

	model = SVC(class_weight='auto', C=1.1000000000000001, gamma=0.10000000000000001) #90
	# model = SVC(class_weight='auto', C=0.80000000000000004, gamma=0.10000000000000001)	#70
	# model = SVC(class_weight='auto', C=12.6, gamma=0.10000000000000001)	#50

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

# used to train each theme
def svm_for_features_fusion(dataset, targets):

	# no feature selection
	model = SVC(class_weight='auto', C=2.5009999999999999, gamma=0.01)	# 0.320476 ; 0.354444	0.375000 ; 0.324444		0.162078 ; 0.439167

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_selected_net(dataset, targets):
	# model = SVC(class_weight='auto', C=1.9, gamma=0.1)	#90
	# model = SVC(class_weight='auto', C=1.3, gamma=0.3)	#70
	# model = SVC(class_weight='auto', C=1.5, gamma=0.1)	#50

	# model = SVC(class_weight='auto', C=0.80000000000000004, gamma=0.30000000000000004) #all

	# model = SVC(class_weight='auto', C=1.6, gamma=3.7) # 90

	model = SVC(class_weight='auto', C=0.9, gamma=3.3) # 90

	model.fit(dataset, targets)
	return model

def svm_selected_ill(dataset, targets):
	# model = SVC(class_weight='auto', C=6.9, gamma=0.7)	#90
	# model = SVC(class_weight='auto', C=1.5, gamma=0.1)	#70
	# model = SVC(class_weight='auto', C=7.1, gamma=0.3)	#50

	# model = SVC(class_weight='auto', C=0.80000000000000004, gamma=0.10000000000000001)	#all

	model = SVC(class_weight='auto', C=2.2, gamma=5.2) # 90

	model.fit(dataset, targets)
	return model

def svm_selected_ideo(dataset, targets):
	# model = SVC(class_weight='auto', C=1.5, gamma=0.1)	#90
	# model = SVC(class_weight='auto', C=0.5, gamma=0.1)	#70
	# model = SVC(class_weight='auto', C=0.5, gamma=0.1)	#50

	# model = SVC(class_weight='auto', C=0.90000000000000002, gamma=0.10000000000000001)	#all

	# model = SVC(class_weight='auto', C=1, gamma=1.6) # 90

	model = SVC(class_weight='auto', C=0.8, gamma=1.6) # 90

	model.fit(dataset, targets)
	return model	






def svm_selected_for_features_fusion(dataset, targets):
	# model = SVC(class_weight='auto', C=0.900000, gamma=0.100000)	#90 	0.066667 ; 0.365278		0.230000 ; 0.268333		0.160087 ; 0.452778
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
	# model = SVC(class_weight='auto', C=0.100000, gamma=0.100000)	#90
	# model = SVC(class_weight='auto', C=0.100000, gamma=0.100000)	#70
	# model = SVC(class_weight='auto', C=0.700000, gamma=0.100000)	#50


	# model = SVC(class_weight='auto', C=0.1, gamma=0.1)	#net
	# model = SVC(class_weight='auto', C=0.1, gamma=0.1)	#ideo

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model