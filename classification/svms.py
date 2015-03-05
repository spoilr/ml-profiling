from sklearn.svm import SVC

def svm_all_vars(dataset, targets):
	# model = SVC(class_weight='auto', C=2.0009999999999999, gamma=0.001) # 0.290895 ; 0.447778
	# model = SVC(class_weight='auto') 									  # 0.244444 ; 0.314444
	model = SVC(class_weight='auto', C=2.0009999999999999)  		      # 0.220000 ; 0.292222

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

def svm_selected_vars(dataset, targets):
	model = SVC(class_weight='auto', C=0.60000999999999993, gamma=0.30001) 	 # 90    0.326667 ; 0.328056
	# model = SVC(class_weight='auto')										 	 # 90    0.381905 ; 0.398056
	# model = SVC(class_weight='auto', C=0.60000999999999993)				 	 # 90    0.332857 ; 0.398056

	# model = SVC(class_weight='auto', gamma=0.001)							 # 70    0.384615 ; 0.596667
	# model = SVC(class_weight='auto')										 # 70    0.328095 ; 0.396667
	# model = SVC(class_weight='auto', C=0.801) 	 				 		 # 70    0.351429 ; 0.383056
	# model = SVC(class_weight='auto', C=0.701)								 # 70    0.351429 ; 0.383056

	# model = SVC(class_weight='auto') 										 # 50    0.351429 ; 0.383056
	# model = SVC(class_weight='auto', gamma=0.001) 				 		 # 50    0.384615 ; 0.596667
	# model = SVC(class_weight='auto', C=0.701)								 # 50    0.327619 ; 0.383056
	# model = SVC(class_weight='auto', C=0.60000999999999993)				 # 50    0.327619 ; 0.383056


	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

# used to train each theme
def svm_for_features_fusion(dataset, targets):
	# model = SVC(class_weight='auto', C=2.5009999999999999, gamma=0.001)   # 0.324127 ; 0.506111    0.441905 ; 0.355556    0.254408 ; 0.398056
	# model = SVC(class_weight='auto') 									    # 0.295000 ; 0.328611    0.345000 ; 0.283611    0.275144 ; 0.402222
	# model = SVC(class_weight='auto', C=2.5009999999999999)			    # 0.243333 ; 0.302500    0.303810 ; 0.271111    0.092308 ; 0.383333

	model = SVC(class_weight='auto', C=0.7, gamma=0.1)	    	# 90	  0.386667 ; 0.330833    0.475238 ; 0.256111    0.385238 ; 0.374444
	# model = SVC(class_weight='auto', C=0.7, gamma=0.2)			# 90	  0.276667 ; 0.318333    0.399048 ; 0.250000    0.000000 ; 0.326944

	# model = SVC(class_weight='auto', C=0.7, gamma=0.1)			# 70	  0.364444 ; 0.329444    0.388254 ; 0.306944    0.328810 ; 0.408056
	# model = SVC(class_weight='auto', C=0.7, gamma=0.2)			# 70	  0.213333 ; 0.338333    0.353333 ; 0.258611    0.086154 ; 0.376667

	# model = SVC(class_weight='auto', C=0.7, gamma=0.1)			# 50	  0.213333 ; 0.341944    0.280000 ; 0.272222    0.080000 ; 0.348056
	# model = SVC(class_weight='auto', C=0.7, gamma=0.3)			# 50	 0.050000 ; 0.325556    0.050000 ; 0.303333    0.207308 ; 0.483333

	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model

# used for selecting the features
def svm_subset_features(dataset, targets):
	# model = SVC(class_weight='auto', C=2.101, gamma=0.001)  		 # 90   0.073333 ; 0.421944     0.033333 ; 0.366944     0.405937 ; 0.545000

	# model = SVC(class_weight='auto', C=0.801, gamma=0.201)		 # 70   0.057143 ; 0.420556     0.097143 ; 0.365833     0.403477 ; 0.543611

	# model = SVC(class_weight='auto', C=0.801, gamma=0.101)	 	 # 50   0.066667 ; 0.386944     0.066667 ; 0.323333     0.420800 ; 0.522778

	model = SVC(class_weight='auto', C=2.8, gamma=0.9)				 # 90	 
	# model = SVC(class_weight='auto', C=0.4, gamma=0.3)				 # 90	 

	# model = SVC(class_weight='auto', C=2.8, gamma=0.9)				 # 70	 
	# model = SVC(class_weight='auto', C=0.1, gamma=0.1)				 # 70	 

	# model = SVC(class_weight='auto', C=2.8, gamma=0.9)				 # 50	 
	# model = SVC(class_weight='auto', C=0.1, gamma=0.1)				 # 50	 


	model.fit(dataset, targets)
	# print 'Model score: %f' % model.score(known_dataset, known_targets)
	return model