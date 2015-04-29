"""
K Nearest Neighbour Classification
Combine KNN for themes
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import *
from fusion import cv10
from fusion import knn
from thematic_data_combined import *

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	tdc = ThematicDataCombined(targets)
	dataset, targets = tdc.thematic_split() 

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	cv10(dataset, targets, fusion_algorithm, ids, knn)

	
