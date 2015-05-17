import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import highval_binary_all_data
from project_data import highval_binary_ideo_data
from project_data import highval_binary_ill_data
from project_data import highval_binary_net_data

from project_data import civil_binary_all_data
from project_data import civil_binary_ideo_data
from project_data import civil_binary_ill_data
from project_data import civil_binary_net_data

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
	nr_features = len(spreadsheet.features) + 1
	assert sum(targets) == len(targets) # all 1s for SPSS

	# append targets to data for SPSS - targests are all 1s
	data = np.hstack((data, np.ones((data.shape[0], 1))))

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
	files = [highval_binary_all_data, highval_binary_ideo_data, highval_binary_ill_data, highval_binary_net_data, civil_binary_all_data, civil_binary_ideo_data, civil_binary_ill_data, civil_binary_net_data]
	for f in files:
		load_and_save(f)





	

	