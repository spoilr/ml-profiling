''' Plot bayesian networks results when considering all nodes, not just the connected ones '''

import numpy as np
import matplotlib.pyplot as plt

net = [(1, 0.6071428571428571), (2, 0.6642857142857141), (3, 0.6642857142857141), (4, 0.6642857142857144), (5, 0.6642857142857144), (6, 0.6500000000000001), (7, 0.6357142857142858), (8, 0.5642857142857143)]
ill = [(1, 0.6461538461538462), (2, 0.7076923076923077), (3, 0.7076923076923077), (4, 0.7076923076923077), (5, 0.7076923076923077), (6, 0.6307692307692309), (7, 0.6307692307692309), (8, 0.6307692307692309)]
ideo = [(1, 0.7307692307692308), (2, 0.7461538461538462), (3, 0.7461538461538462), (4, 0.7461538461538462)]

def unique(lst):
	max_vals = dict()
	for pair in lst:
		if pair[1] >= 0.6 and (pair[0] not in max_vals.keys() or pair[1] > max_vals[pair[0]]):
			max_vals[pair[0]] = pair[1]
	vals = []
	for key, value in max_vals.iteritems():
	    temp = (key,value)
	    vals.append(temp)		    
	return vals

if __name__ == "__main__":

	net = set(net)
	ill = set(ill)
	ideo = set(ideo)

	net = unique(net)
	ill = unique(ill)
	ideo = unique(ideo)

	plt.plot([t[0] for t in net], [t[1] for t in net], 'ro-', [t[0] for t in ill], [t[1] for t in ill], 'bo-', [t[0] for t in ideo], [t[1] for t in ideo], 'go-')
	# plt.axis([0, 20, 0, 1])
	plt.xticks(np.arange(0, 10, 1.0))
	plt.show()