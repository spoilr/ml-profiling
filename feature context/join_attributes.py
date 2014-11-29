print(__doc__)

import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *
from split_dataset import *
import itertools

import operator
import numpy as np
import matplotlib.pyplot as plt

def join_attributes(attr1, attr2):
	iterables = [set(attr1.tolist()), set(attr2.tolist())]
	for t in itertools.product(*iterables):
		print t

if __name__ == "__main__":
	spreadsheet = Spreadsheet('/home/user/Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme(sys.argv[1])
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)

	join_attributes(known_dataset[:,0], known_dataset[:,1])