from pgml import *
from create_struct import *
from join_tree import *
from max_learning_data import max_learning_data
import operator
from collections import OrderedDict
from copy import deepcopy
from itertools import chain		

def is_path(adj_list, node1, node2, visited=[]):
    visited = visited + [node1]
    if node1 == node2:
        return True
    for node in adj_list[node1]:
        if node not in visited:
            path = is_path(adj_list, node, node2, visited)
            if path:
                return path
    return False    

def create_adjacency_list(nodes):
    adj_list = dict()
    for key in nodes:
        adj_list[key] = []
    return adj_list     

def create_graph(edges, nodes):
    final_edges = []
    adj_list = create_adjacency_list(nodes)
    for x in edges:
        node1 = x[0]
        node2 = x[1]
        if not is_path(adj_list, node2, node1):
            adj_list[node1].append(node2)
            final_edges.append([node1, node2])
    return final_edges

def set_of_edges(edges):
	unique_edges = []
	for edge in edges:
		if edge not in unique_edges:
			unique_edges.append(edge)
	return unique_edges			

def combine_network(bn_net, bn_ill, bn_ideo):
	nodes = []
	edges = []
	cpts = dict()

	nodes += bn_ideo.V
	edges += bn_ideo.E
	nodes += bn_ill.V
	edges += bn_ill.E
	nodes += bn_net.V
	edges += bn_net.E
	nodes = set(nodes)

	edges = set_of_edges(create_graph(edges, nodes)))
	
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

	evidence = dict(InteractNet="0", MentalIll="1")	### NEEDS TO BE STRING
	inf = jt_inference(bn, evidence)
	print inf