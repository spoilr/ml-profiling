from sklearn.svm import SVC

def svm_all_vars(dataset, targets):
	model = SVC(class_weight='auto', C=2.0009999999999999, gamma=0.001) # 0.290895 ; 0.447778
	# model = SVC(class_weight='auto') 									# 0.244444 ; 0.314444
	# model = SVC(class_weight='auto', C=2.0009999999999999)  			# 0.220000 ; 0.292222

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_selected_vars(dataset, targets):
	model = SVC(class_weight='auto', C=0.60000999999999993, gamma=0.30001) # 90    0.384615 ; 0.596667
	# model = SVC(class_weight='auto')										 # 90    0.290476 ; 0.550833
	# model = SVC(class_weight='auto', C=0.60000999999999993)				 # 90    0.263810 ; 0.560556

	# model = SVC(class_weight='auto', C=0.101, gamma=0.001) 				 # 70    0.464615 ; 0.696667
	# model = SVC(class_weight='auto', gamma=0.001)							 # 70    0.384615 ; 0.596667
	# model = SVC(class_weight='auto')										 # 70    0.306349 ; 0.540833
	# model = SVC(class_weight='auto', C=0.801) 	 				 		 # 70    0.332619 ; 0.530833
	# model = SVC(class_weight='auto', C=0.701)								 # 70    0.316688 ; 0.531944

	# model = SVC(class_weight='auto', C=0.101, gamma=0.001) 				 # 50    0.474615 ; 0.671667
	# model = SVC(class_weight='auto') 										 # 50    0.342561 ; 0.509444    
	# model = SVC(class_weight='auto', gamma=0.001) 				 		 # 50    0.384615 ; 0.596667
	# model = SVC(class_weight='auto', C=0.701)								 # 50    0.289784 ; 0.539444
	# model = SVC(class_weight='auto', C=0.60000999999999993)				 # 50    0.300895 ; 0.536944
	# model = SVC(class_weight='auto', C=0.801) 	 				 		 # 50    0.318355 ; 0.529444


	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

# used to train each theme
def svm_for_features_fusion(dataset, targets):
	model = SVC(class_weight='auto', C=2.5009999999999999, gamma=0.001)   # 0.324127 ; 0.506111    0.441905 ; 0.355556    0.275144 ; 0.402222
	# model = SVC(class_weight='auto') 									  # 0.295000 ; 0.328611    0.345000 ; 0.283611    0.275144 ; 0.402222
	# model = SVC(class_weight='auto', C=2.5009999999999999)			  # 0.243333 ; 0.302500    0.303810 ; 0.271111    0.275144 ; 0.402222	

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

# used for selecting the features
def svm_subset_features(dataset, targets):
	model = SVC(class_weight='auto', C=2.101, gamma=0.001)  		 # 90   0.388462 ; 0.586667    0.396154 ; 0.566667    0.405937 ; 0.545000
	# model = SVC(class_weight='auto', C=2.101) 					 # 90
	# model = SVC(class_weight='auto', gamma=0.001) 				 # 90

	# model = SVC(class_weight='auto', C=0.801, gamma=0.201)	 	 # 70
	# model = SVC(class_weight='auto')						 		 # 70
	# model = SVC(class_weight='auto', C=0.801)	 			 		 # 70
	# model = SVC(class_weight='auto', gamma=0.201)	 		 		 # 70
	# model = SVC(class_weight='auto', C=0.801, gamma=0.101)	 	 # 70
	# model = SVC(class_weight='auto', C=0.801)	 					 # 70
	# model = SVC(class_weight='auto', gamma=0.101)	 				 # 70
	# model = SVC(class_weight='auto', C=0.101, gamma=0.001)		 # 70
	# model = SVC(class_weight='auto', C=0.101)				 		 # 70
	# model = SVC(class_weight='auto', gamma=0.001)	 		 		 # 70

	# model = SVC(class_weight='auto')						 		 # 50
	# model = SVC(class_weight='auto', C=0.801)	 			 		 # 50
	# model = SVC(class_weight='auto', gamma=0.201)	 		 		 # 50
	# model = SVC(class_weight='auto', C=0.801, gamma=0.101)	 	 # 50
	# model = SVC(class_weight='auto', C=0.801)	 			 		 # 50
	# model = SVC(class_weight='auto', gamma=0.101)	 		 		 # 50
	# model = SVC(class_weight='auto', C=0.101, gamma=0.001)	 	 # 50
	# model = SVC(class_weight='auto', C=0.101)				 		 # 50
	# model = SVC(class_weight='auto', gamma=0.001)	 		 		 # 50



	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model