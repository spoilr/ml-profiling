"""
SVM C-Support Vector Classification
"""

print(__doc__)


import sys
sys.path.insert(0, 'utils/')
from load_data import *
from parse_theme import *

from sklearn.svm import SVC

# splits dataset into 2 sets: one for which we know the target 
# and one for which the target is unknown
def split_dataset(dataset, targets):
	unknowns = []
	known_dataset = []
	known_targets = []
	for i in range(0, len(targets)):
		if targets[i] == 0:
			unknowns.append(dataset[i])
		else:	
			known_dataset.append(dataset[i])
			known_targets.append(targets[i])

	return [np.array(known_dataset), known_targets, np.array(unknowns)]	

def svm(dataset, targets):
	[known_dataset, known_targets, unknowns, ] = split_dataset(dataset, targets)

	model = SVC(kernel='linear')
	model.fit(known_dataset, known_targets)
	print 'Model score: %f' % model.score(known_dataset, known_targets)

if __name__ == "__main__":
	spreadsheet = Spreadsheet('/home/user/Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
	targets = data.targets

	[dataset, features] = parse_theme(sys.argv[1])
	svm(dataset, targets)