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
	key_dict = dict()
	isolated = []

	for x in Vdata.iteritems():
		key = x[0]
		parents = x[1]['parents']
		children = x[1]['children']
		values = x[1]['vals']
		cprob = x[1]['cprob']
		values = map(str, values)
		
		key_dict[key] = {}
		if not parents and not children:
			key_dict[key] = dict(zip(values, cprob))
			isolated.append(key)

		if parents:
			cpts[str(key)] = create_cpts_from_parents(values, parents, cprob)
		elif children:
			cpts[str(key)] = [[[], dict(zip(values, cprob))]]

	return build_bbn_from_conditionals(cpts), key_dict, isolated
	
def jt_inference(network, evidence):
	bn, inf, isolated = create_bbn_network(network.Vdata)		
	result = bn.query(**evidence)
	for (key, val), prob in result.iteritems():
		inf[key][val] = prob

	keys = evidence.keys()
	for key in keys:
		if key in isolated:
			ev = evidence[key]
			for k, v in inf[key].iteritems():
				if k == ev:
					inf[key][k] = 1
				else:
					inf[key][k] = 0	
			
	return inf	

if __name__ == "__main__":
	bn_net = create_bayesian_network_structure('net')
	evidence = dict(InteractNet="0", Getaway="1")	### NEEDS TO BE STRING
	inf = jt_inference(bn_net, evidence)
	print json.dumps(inf, indent=2)
