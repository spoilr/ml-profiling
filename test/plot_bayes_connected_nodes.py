''' Plot bayesian networks results when considering all nodes, not just the connected ones '''

import numpy as np
import matplotlib.pyplot as plt

# all yes
# net = [(11, 0.5642857142857143), (10, 0.5785714285714285), (9, 0.5928571428571429), (8, 0.6071428571428571), (7, 0.6357142857142858), (6, 0.65), (5, 0.6642857142857144), (4, 0.65), (3, 0.6071428571428571), (2, 0.6071428571428571), (1, 0.6071428571428571)] 
# ill = [(11, 0.5153846153846154), (10, 0.4153846153846154), (9, 0.49230769230769234), (8, 0.5538461538461539), (7, 0.5846153846153845), (6, 0.6307692307692309), (5, 0.6461538461538462), (4, 0.6615384615384615), (3, 0.6615384615384616), (2, 0.7076923076923077), (1, 0.6461538461538462)]
# ideo = [(11, 0.5153846153846154), (10, 0.5923076923076923), (9, 0.6692307692307692), (8, 0.6846153846153846), (7, 0.7153846153846153), (6, 0.7307692307692308), (5, 0.7307692307692308), (4, 0.7461538461538462), (3, 0.7461538461538462), (2, 0.7461538461538462), (1, 0.7307692307692308)]

# all no
# net = [(13, 0.7214285714285714), (12, 0.7214285714285714), (11, 0.7214285714285714), (10, 0.7214285714285714), (9, 0.7214285714285714), (8, 0.7214285714285714), (7, 0.7214285714285714), (6, 0.7214285714285714), (5, 0.7214285714285714), (4, 0.7214285714285714), (3, 0.7071428571428573), (2, 0.6785714285714286), (1, 0.6642857142857143)]
# ill = [(13, 0.8307692307692308), (12, 0.8307692307692308), (11, 0.8307692307692308), (10, 0.8307692307692308), (9, 0.8307692307692308), (8, 0.8307692307692308), (7, 0.8307692307692308), (6, 0.8307692307692308), (5, 0.8307692307692308), (4, 0.8), (3, 0.8), (2, 0.7538461538461538), (1, 0.7384615384615385)]
# ideo = [(13, 0.7923076923076924), (12, 0.7692307692307692), (11, 0.7923076923076924), (10, 0.7923076923076924), (9, 0.7923076923076924), (8, 0.7923076923076924), (7, 0.7923076923076924), (6, 0.7923076923076924), (5, 0.7923076923076924), (4, 0.7923076923076924), (3, 0.7923076923076924), (2, 0.7769230769230769), (1, 0.7538461538461537)]

# crim yes
# net = [(6, 0.5357142857142858), (5, 0.5785714285714285), (4, 0.6071428571428571), (3, 0.65), (2, 0.6071428571428571), (1, 0.6071428571428571)]
# ill = [(6, 0.4615384615384615), (5, 0.5384615384615385), (4, 0.5846153846153846), (3, 0.6307692307692309), (2, 0.6615384615384615), (1, 0.6461538461538462)]
# ideo = [(6, 0.6076923076923076), (5, 0.6692307692307693), (4, 0.7), (3, 0.7), (2, 0.7153846153846154), (1, 0.7076923076923077)]

# crim no
# net = [(6, 0.6500000000000001), (5, 0.7071428571428573), (4, 0.7071428571428571), (3, 0.6642857142857144), (2, 0.6499999999999999), (1, 0.6499999999999999)]
# ill = [(6, 0.7384615384615385), (5, 0.7384615384615385), (4, 0.7384615384615385), (3, 0.7384615384615385), (2, 0.7384615384615385), (1, 0.6923076923076924)]
# ideo = [(6, 0.7461538461538461), (5, 0.7923076923076924), (4, 0.7923076923076924), (3, 0.7923076923076924), (2, 0.7538461538461537), (1, 0.7384615384615386)]




# all yes 0.6
net = [(11, 0.5230769230769231), (10, 0.5666666666666667), (9, 0.5916666666666667), (8, 0.6), (7, 0.5545454545454546), (6, 0.57), (5, 0.6545454545454545), (4, 0.6799999999999999), (3, 0.6642857142857144), (2, 0.6357142857142858), (1, 0.6357142857142858)]
ill = [(11, 0.4), (10, 0.4153846153846154), (9, 0.4615384615384615), (8, 0.5076923076923078), (7, 0.55), (6, 0.5727272727272728), (5, 0.6636363636363637), (4, 0.6636363636363637), (3, 0.7384615384615384), (2, 0.6923076923076923), (1, 0.6461538461538462)]
ideo = [(11, 0.5153846153846154), (10, 0.5923076923076923), (9, 0.6461538461538462), (8, 0.6916666666666667), (7, 0.6181818181818183), (6, 0.6636363636363637), (5, 0.76), (4, 0.7899999999999999), (3, 0.7769230769230769), (2, 0.7461538461538462), (1, 0.7000000000000001)]


# all no 0.6
net = [(13, 0.7214285714285713), (12, 0.7214285714285714), (11, 0.7214285714285714), (10, 0.7214285714285714), (9, 0.7214285714285714), (8, 0.7214285714285714), (7, 0.7214285714285715), (6, 0.7214285714285715), (5, 0.7214285714285715), (4, 0.7214285714285714), (3, 0.7071428571428572), (2, 0.7071428571428571), (1, 0.65)]
ill = [(13, 0.8307692307692308), (12, 0.8307692307692308), (11, 0.8307692307692308), (10, 0.8307692307692308), (9, 0.8307692307692308), (8, 0.8307692307692308), (7, 0.8307692307692308), (6, 0.8307692307692308), (5, 0.8307692307692308), (4, 0.8307692307692308), (3, 0.8), (2, 0.7846153846153846), (1, 0.7384615384615385)]
ideo = [(13, 0.7769230769230769), (12, 0.7923076923076924), (11, 0.7923076923076924), (10, 0.7923076923076924), (9, 0.7923076923076924), (8, 0.7923076923076924), (7, 0.7923076923076924), (6, 0.7923076923076924), (5, 0.7923076923076924), (4, 0.7923076923076924), (3, 0.7923076923076924), (2, 0.7769230769230769), (1, 0.6615384615384616)]

# crim yes 0.6
net = [(6, 0.40714285714285714), (5, 0.5785714285714285), (4, 0.6071428571428572), (3, 0.6071428571428572), (2, 0.6071428571428571), (1, 0.5214285714285716)]
ill = [(6, 0.4), (5, 0.5692307692307692), (4, 0.5692307692307692), (3, 0.5846153846153845), (2, 0.7230769230769231), (1, 0.7076923076923077)]
ideo = [(6, 0.5461538461538462), (5, 0.6461538461538462), (4, 0.5769230769230769), (3, 0.6923076923076924), (2, 0.6615384615384615), (1, 0.676923076923077)]

# crim no 0.6
net = [(6, 0.5071428571428571), (5, 0.6642857142857144), (4, 0.6500000000000001), (3, 0.7071428571428571), (2, 0.6928571428571431), (1, 0.5857142857142856)]
ill = [(6, 0.5384615384615385), (5, 0.6923076923076924), (4, 0.8), (3, 0.8307692307692308), (2, 0.723076923076923), (1, 0.7846153846153846)]
ideo = [(6, 0.7230769230769231), (5, 0.7384615384615384), (4, 0.7076923076923076), (3, 0.7384615384615386), (2, 0.7153846153846154), (1, 0.5923076923076923)]

def unique(lst):
	max_vals = dict()
	for pair in lst:
		if pair[1] >= 0.4 and (pair[0] not in max_vals.keys() or pair[1] > max_vals[pair[0]]):
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

	gnet, = plt.plot([t[0] for t in net], [t[1] for t in net], 'ro-')
	gill, = plt.plot([t[0] for t in ill], [t[1] for t in ill], 'bo-')
	gideo, = plt.plot([t[0] for t in ideo], [t[1] for t in ideo], 'go-')
	plt.legend([gnet, gill, gideo], ['net', 'ill', 'ideo'])
	plt.axis([0, 7, 0.4, 1])
	plt.xticks(np.arange(0, 7, 1.0))
	plt.yticks(np.arange(0.4, 1, 0.05))
	

	plt.show()