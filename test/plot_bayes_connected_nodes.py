''' Plot bayesian networks results when considering all nodes, not just the connected ones '''

import numpy as np
import matplotlib.pyplot as plt

# net = [(0, 0.607143), (1, 0.607143), (2, 0.607143), (3, 0.607143), (4, 0.607143)]
# ill = [(4, 0.661538), (5, 0.661538), (5, 0.692308)]
# ideo = [(0, 0.707692), (1, 0.707692), (2, 0.707692), (3, 0.707692), (4, 0.707692), (5, 0.707692), (1, 0.730769), (2, 0.730769), (3, 0.730769), (4, 0.730769), (5, 0.730769), (1, 0.715385), (2, 0.715385), (3, 0.715385), (4, 0.715385), (5, 0.715385), (2, 0.723077), (3, 0.723077), (4, 0.723077), (5, 0.723077), (1, 0.746154), (2, 0.746154), (3, 0.746154), (4, 0.746154), (5, 0.746154)]

net = [(1, 0.6071428571428571), (2, 0.6642857142857141), (3, 0.6642857142857141), (4, 0.6642857142857144), (5, 0.6642857142857144)]
ill = [(1, 0.6461538461538462), (2, 0.7076923076923077), (3, 0.7076923076923077), (4, 0.7076923076923077), (5, 0.7076923076923077)]
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
	plt.xticks(np.arange(0, 8, 1.0))
	plt.show()