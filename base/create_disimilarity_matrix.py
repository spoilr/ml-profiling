import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import binary_all
from project_data import binary_ideo
from project_data import binary_ill
from project_data import binary_net

import numpy as np
import csv
from sklearn.metrics import jaccard_similarity_score

''' The last column represents the targets, but for SPSS all are 1s as we are looking at highvalue and civil respectively.'''

def create_matrix(data, nr_features):
	disimilarity_matrix = np.zeros((nr_features, nr_features))
	for i in range(nr_features):
		for j in range(nr_features):
			disimilarity_matrix[i][j] = 1 - round(jaccard_similarity_score(data[:,i], data[:,j]), 3)
	return disimilarity_matrix		

def load_and_save(data_file):
	spreadsheet = Spreadsheet(data_file, upsampling=False)
	data = Data(spreadsheet, upsampling=False)
	targets = data.targets
	data = data.examples
	nr_features = len(spreadsheet.features) + 2

	val_civil = targets
	val_highvalue = [1-x for x in targets]

	print data.shape

	data = np.hstack((data, np.array(val_civil).reshape(-1, 1)))
	data = np.hstack((data, np.array(val_highvalue).reshape(-1, 1)))

	print data.shape
	disimilarity_matrix = create_matrix(data, nr_features)
	assert disimilarity_matrix.shape == (nr_features, nr_features)

	file_name = str(data_file).split('/')[-1].split('.')[0] + '.csv'

	fl = open(file_name, 'w')
	writer = csv.writer(fl)
	for values in disimilarity_matrix:
	    writer.writerow(values)
	fl.close()

if __name__ == "__main__":
	files = [binary_all, binary_ideo, binary_ill, binary_net]
	for f in files:
		load_and_save(f)





	

	