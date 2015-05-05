from create_struct import *
from bayesian.bbn import *
import json

def create_parents(keys, parents):
	pars = []
	for i in range(len(parents)):
		pars.append([parents[i], str(keys[i])])
	return pars	

def create_cpts_from_parents(values, parents, cprob):
	result = []
	for keys, prob_vals in cprob.iteritems():
		keys = eval(keys)
		result.append([create_parents(keys, parents), dict(zip(values, prob_vals))])
	return result

def create_bbn_network(Vdata):
	cpts = dict()
	for x in Vdata.iteritems():
		key = x[0]
		parents = x[1]['parents']
		children = x[1]['children']
		values = x[1]['vals']
		cprob = x[1]['cprob']
		values = map(str, values)

		if parents:
			cpts[str(key)] = create_cpts_from_parents(values, parents, cprob)
		elif children:
			cpts[str(key)] = [[[], dict(zip(values, cprob))]]

	node_funcs = []
	domains = dict()
	for variable_name, cond_tt in cpts.items():
		node_func = make_node_func(variable_name, cond_tt)
		node_funcs.append(node_func)
		domains[variable_name] = node_func._domain

	return build_bbn_from_conditionals(cpts)
	
if __name__ == "__main__":
	bn_net = create_bayesian_network_structure('ideo')
	bn = create_bbn_network(bn_net.Vdata)		
	result = bn.query()
	print result








