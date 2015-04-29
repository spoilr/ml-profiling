"""
Decision Tree Classification
Single DT
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import *
from parse_theme import *
from split_dataset import *
from cv import cv10
from cv import dt_one_fold_measures

import numpy as np

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		[known_dataset, known_targets, unk] = split_dataset(dataset, targets)

		cv10(np.array(known_dataset), np.array(known_targets), ids, dt_one_fold_measures)
		
	except IndexError:
		print "Error!! Pass 'all' as argument"