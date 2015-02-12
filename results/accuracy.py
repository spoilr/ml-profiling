'''
	SVM results with accuracy
	For combined SVMs, need results for each of the methods used for combination: Majority, Weighted Majority, SVM,
'''

svm_all_vars = [
SVC() # 0.303333
SVC(class_weight='auto') # 0.325833

SVC(class_weight='auto', C=0.90000999999999987, gamma=0.30001) # 0.303333
SVC(C=0.90000999999999987, gamma=0.30001) # 0.303333
SVC(class_weight='auto', gamma=0.30001) # 0.303333
SVC(class_weight='auto', C=0.90000999999999987) # 0.325833
SVC(C=0.90000999999999987) # 0.303333
SVC(gamma=0.30001) # 0.303333

SVC(class_weight='auto', C=0.9, gamma=0.303333) # 0.303333
SVC(C=0.9, gamma=0.303333) # 0.303333
SVC(class_weight='auto', gamma=0.303333) # 0.303333
SVC(class_weight='auto', C=0.9) # 0.325833
SVC(C=0.9) # 0.303333
SVC(gamma=0.303333) # 0.303333

SVC(class_weight='auto', C=0.801000, gamma=0.101000) # 0.303333
SVC(C=0.801000, gamma=0.101000) # 0.303333
SVC(class_weight='auto', gamma=0.101000) # 0.303333
SVC(class_weight='auto', C=0.801000) # 0.349444
SVC(gamma=0.101000) # 0.303333
SVC(C=0.801000) # 0.303333
]

svm_selected_vars = [
SVC() # 0.303333
SVC(class_weight='auto') # 0.395000

SVC(class_weight='auto', C=0.90000999999999987, gamma=12.600009999999999) # 0.313333
SVC(C=0.90000999999999987, gamma=12.600009999999999) # 0.303333
SVC(class_weight='auto', gamma=12.600009999999999) # 0.313333
SVC(class_weight='auto', C=0.90000999999999987) # 0.405000
SVC(C=0.90000999999999987) # 0.303333
SVC(gamma=12.600009999999999) # 0.303333

SVC(class_weight='auto', C=0.9, gamma=14.700010) # 0.303333
SVC(C=0.9, gamma=14.700010) # 0.303333
SVC(class_weight='auto', gamma=14.700010) # 0.303333
SVC(class_weight='auto', C=0.9) # 0.405000
SVC(C=0.9) # 0.303333
SVC(gamma=14.700010) # 0.303333

SVC(class_weight='auto', C=0.901000, gamma=14.401000) # 0.303333
SVC(C=0.901000, gamma=14.401000) # 0.303333
SVC(class_weight='auto', gamma=14.401000) # 0.303333
SVC(class_weight='auto', C=0.901000) # 0.405000
SVC(C=0.901000) # 0.303333
SVC(gamma=14.401000) # 0.303333
]

svm_for_features_fusion = [
SVC(class_weight='auto', C=0.90000999999999987, gamma=0.30001) # 0.303333
]

svm_subset_features = [
SVC()
SVC(class_weight='auto')

SVC(class_weight='auto', C=0.90000999999999987, gamma=0.30001)
SVC(C=0.90000999999999987, gamma=0.30001)
SVC(class_weight='auto', gamma=0.30001)
SVC(class_weight='auto', C=0.90000999999999987)
SVC(C=0.90000999999999987)
SVC(C=0.90000999999999987)

SVC(class_weight='auto', C=0.801000, gamma=0.201000)
SVC(C=0.801000, gamma=0.201000)
SVC(class_weight='auto', gamma=0.201000)
SVC(class_weight='auto', C=0.801000)
SVC(gamma=0.201000)
SVC(C=0.801000)
]
	