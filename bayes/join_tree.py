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

		if parents:
			cpts[str(key)] = create_cpts_from_parents(values, parents, cprob)
		elif children:
			cpts[str(key)] = [[[], dict(zip(values, cprob))]]

	return build_bbn_from_conditionals(cpts), key_dict
	
def inference(network):
	bn, inf = create_bbn_network(network.Vdata)		
	result = bn.query()
	for (key, val), prob in result.iteritems():
		inf[key][val] = prob
	return json.dumps(inf, indent=2)

if __name__ == "__main__":
	bn_net = create_bayesian_network_structure('ideo')
	bn, key_dict = create_bbn_network(bn_net.Vdata)		
	result = bn.query()
	print result
	print inference(bn_net)








