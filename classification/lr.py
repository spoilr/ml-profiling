import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import *
from load_data import *
from parse_theme import *
from split_dataset import *
from lr_fusion import cross_validation
from thematic_data_combined import *

import numpy as np

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	tdc = ThematicDataCombined(targets)
	dataset, targets = tdc.thematic_split() 

	fusion_algorithm = raw_input("Enter algorithm. Choose between maj, wmaj, svm, nn")
	cross_validation(dataset, targets, fusion_algorithm, ids)

	
