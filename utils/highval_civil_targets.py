from load_data import *
import numpy as np

def extract_high_val_civil_targets(examples):
	np_examples = np.array(examples)
	(rows, cols) = np_examples.shape
	return np_examples[:, cols - 1]


if __name__ == "__main__":
	spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)

	high_val_civil_targets = extract_high_val_civil_targets(data.examples)

	for s in high_val_civil_targets:
		print s