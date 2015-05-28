from create_struct import *
from join_tree import *
from max_learning_data import max_learning_data
import operator
from collections import OrderedDict
from copy import deepcopy

def combine_network(bn_net, bn_ill, bn_ideo):
	nodes = []
	nodes += bn_net.V
	nodes += bn_ill.V
	nodes += bn_ideo.V
	nodes = list(set(nodes))

	all_edges = []
	all_edges += bn_net.E
	all_edges += bn_ill.E
	all_edges += bn_ideo.E
	edges = []
	for edge in all_edges:
		if edge not in edges:
			edges.append(edge)

	skel = GraphSkeleton()
  	skel.load_skel(nodes, edges)
  	skel.toporder()

  	learner = PGMLearner()
  	bn, passed = learner.discrete_mle_estimateparams(skel, max_learning_data)
  	return bn

if __name__ == "__main__":
	bn_net = create_bayesian_network_structure('net')
	bn_ill = create_bayesian_network_structure('ill')
	bn_ideo = create_bayesian_network_structure('ideo')
	bn = combine_network(bn_net, bn_ill, bn_ideo)
	print len(bn.V)
	print len(bn.E)