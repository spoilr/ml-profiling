import sys
sys.path.insert(0, 'utils/')
from load_data import *
from project_data import *
from parse_theme import *
from split_dataset import *

import numpy as np

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme('all')
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)

	print 'POSITIVE %d' % len([x for x in known_targets if x==1])
	print 'NEGATIVE %d' % len([x for x in known_targets if x==0])