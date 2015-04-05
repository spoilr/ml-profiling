svm_all_vars = [
SVC(class_weight='auto', C=2.0009999999999999)  		      # 0.220000 ; 0.292222
SVC(class_weight='auto', C=2.0009999999999999, gamma=0.001)	  # 0.290895 ; 0.447778
SVC(class_weight='auto') 									  # 0.244444 ; 0.314444
SVC(class_weight='auto', C=0.700000, gamma=0.300000)		  # 0.200000 ; 0.436667
SVC(class_weight='auto', C=0.700000, gamma=0.100000)		  # 0.204545 ; 0.425556	
]





svm_for_features_fusion = [
SVC(class_weight='auto', C=2.5009999999999999, gamma=0.001) 	# 0.324127 ; 0.506111	0.441905 ; 0.355556		0.254408 ; 0.398056
SVC(class_weight='auto')										# 0.295000 ; 0.328611	0.345000 ; 0.283611		0.275144 ; 0.402222
SVC(class_weight='auto', C=2.5009999999999999)					# 0.243333 ; 0.302500	0.303810 ; 0.271111		0.092308 ; 0.383333
SVC(class_weight='auto', C=0.700000, gamma=0.100000)			# 0.135714 ; 0.347778	0.182143 ; 0.281111		0.111905 ; 0.395000
SVC(class_weight='auto', C=2.5009999999999999, gamma=0.001)		# 0.324127 ; 0.506111	0.441905 ; 0.355556		0.254408 ; 0.398056
]








svm_for_features_fusion_90 = [
SVC(class_weight='auto', C=1.000000, gamma=0.200000)	#90		0.000000 ; 0.314444		0.263333 ; 0.247500		0.066667 ; 0.311944
SVC(class_weight='auto', C=0.700000, gamma=0.300000)	#90		0.208990 ; 0.400000		0.170000 ; 0.293333		0.207143 ; 0.390000
SVC(class_weight='auto', C=0.700000, gamma=0.100000)	#90		0.100000 ; 0.365278		0.240000 ; 0.277222		0.141905 ; 0.390556
SVC(class_weight='auto', C=0.700000, gamma=0.200000)	#90		0.000000 ; 0.335556		0.166667 ; 0.282222		0.116667 ; 0.367778
SVC(class_weight='auto', C=1.900000, gamma=0.100000)	#90 	0.000000 ; 0.336667		0.183333 ; 0.291944		0.237143 ; 0.334444
SVC(class_weight='auto', C=2.5009999999999999)			#90		0.158333 ; 0.380000		0.370000 ; 0.301944		0.270476 ; 0.409444
]

svm_subset_features_90 = [
SVC(class_weight='auto', C=0.100000, gamma=0.100000)	#90
SVC(class_weight='auto', C=0.100000, gamma=0.500000)	#90
SVC(class_weight='auto', C=0.100000, gamma=0.100000)	#90
SVC(class_weight='auto', C=0.100000, gamma=0.300000)	#90
SVC(class_weight='auto', C=0.100000, gamma=0.300000)	#90
SVC(class_weight='auto', C=0.100000, gamma=0.100000)	#90
]

svm_for_features_fusion_70 = [
SVC(class_weight='auto', C=0.700000, gamma=0.200000)	#70 	0.073333 ; 0.370000		0.120000 ; 0.292222		0.291905 ; 0.406111
]

svm_subset_features_70 = [
SVC(class_weight='auto', C=0.100000, gamma=0.100000)	#70
]

svm_for_features_fusion_50 = [
SVC(class_weight='auto', C=0.700000, gamma=0.200000)		#50 	0.219091 ; 0.392222		0.126667 ; 0.336667		0.000000 ; 0.358889
SVC(class_weight='auto', C=2.5009999999999999)				#50 	0.123333 ; 0.408889 	0.260000 ; 0.293333		0.258990 ; 0.481944
SVC(class_weight='auto', C=2.5009999999999999, gamma=0.001) #50 	0.383217 ; 0.589167		0.387762 ; 0.566667		0.075000 ; 0.342222
]

svm_subset_features_50 = [
SVC(class_weight='auto', C=0.100000, gamma=0.100000)	#50
SVC(class_weight='auto', C=0.700000, gamma=0.100000)	#50
SVC(class_weight='auto', C=0.700000, gamma=0.100000)	#50
]






svm_selected_vars_90 = [
SVC(class_weight='auto', C=0.60000999999999993, gamma=0.30001)	#90 	0.108333 ; 0.453333
SVC(class_weight='auto', C=0.701)								#90 	0.355952 ; 0.479444
SVC(class_weight='auto', C=0.700000, gamma=0.300000)			#90 	0.050000 ; 0.314444
SVC(class_weight='auto', C=0.700000, gamma=0.200000)			#90 	0.134444 ; 0.346667
SVC(class_weight='auto', C=1.900000, gamma=0.100000)			#90 	0.068571 ; 0.432222
]

all_svm_subset_features_90 = [
SVC(class_weight='auto', C=0.7, gamma=0.1)				#90
SVC(class_weight='auto', C=0.7, gamma=0.1)				#90
SVC(class_weight='auto', C=0.7, gamma=0.2)				#90
SVC(class_weight='auto', C=0.1, gamma=0.1)				#90
SVC(class_weight='auto', C=0.100000, gamma=0.500000)	#90
]

svm_selected_vars_70 = [
SVC(class_weight='auto', C=0.700000, gamma=0.100000)		#70 	0.033333 ; 0.454444
SVC(class_weight='auto', C=1.900000, gamma=0.100000)		#70 	0.040000 ; 0.324444
SVC(class_weight='auto', C=0.700000, gamma=0.200000)		#70 	0.040000 ; 0.347778
]

all_svm_subset_features_70 = [
SVC(class_weight='auto', C=0.4, gamma=0.3)				#70
SVC(class_weight='auto', C=0.100000, gamma=0.300000)	#70
SVC(class_weight='auto', C=0.1, gamma=0.1)				#70
]

svm_selected_vars_50 = [
SVC(class_weight='auto')								#50 	0.278333 ; 0.410278
SVC(class_weight='auto', C=0.801)						#50 	0.283571 ; 0.421667
SVC(class_weight='auto', C=0.701)						#50 	0.275238 ; 0.432778
SVC(class_weight='auto', C=0.700000, gamma=0.100000)	#50 	0.173333 ; 0.323056
SVC(class_weight='auto', C=0.700000, gamma=0.200000)	#50 	0.050000 ; 0.303333
SVC(class_weight='auto', C=1.900000, gamma=0.100000)	#50 	0.040000 ; 0.335556
SVC(class_weight='auto', C=2.5009999999999999)			#50 	0.403571 ; 0.376667
]

all_svm_subset_features_50 = [
SVC(class_weight='auto', C=0.1, gamma=0.1)		#50
SVC(class_weight='auto', C=0.7, gamma=0.2)		#50
SVC(class_weight='auto', C=0.4, gamma=0.3)		#50
SVC(class_weight='auto', C=0.4, gamma=0.3)		#50
SVC(class_weight='auto', C=0.1, gamma=0.1)		#50
SVC(class_weight='auto', C=0.4, gamma=0.3)		#50
SVC(class_weight='auto', C=0.4, gamma=0.3)		#50
]