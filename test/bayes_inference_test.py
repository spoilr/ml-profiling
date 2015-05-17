import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'bayes/')
from create_struct import *
from load_data import *
from parse_theme import *
from project_data import *

def inference_accuracy(dataset):
	accuracy = 0
	for instance in dataset:
		accuracy += inference_accuracy_for_instance(bn.V, features, inf, instance)
	return float(accuracy) / len(dataset)	

def inference_accuracy_for_instance(nodes, features, inference, instance):
	nrs = 0
	for key, (val, prob) in inference.iteritems():
		index = features.index(key)
		if instance[index] == int(val):
			nrs+=1

	accuracy = float(nrs)/len(nodes)
	print 'Accuracy ' + str(accuracy)
	return accuracy

def likelihood_from_inference(inference):
	inf = dict()
	inference = json.loads(inference)
	for key, vals in inference.items():
		inf[key] = max(vals.items(), key=lambda x: x[1])
	return inf	

if __name__ == "__main__":
	theme = raw_input("Enter theme.\n")
	bn = create_bayesian_network_structure(theme)
	print 'Nodes ' + str(bn.V)

	spreadsheet = Spreadsheet(addendum_data_file, upsampling=False)
	data = Data(spreadsheet, upsampling=False)
	targets = np.array(data.targets)
	[dataset, features] = parse_theme_from_file(theme, addendum_data_file)

	dataset = np.hstack((dataset, targets.reshape(len(targets), 1))) # append targets
	features.append('HighValueCivilian')	# append target name in feature
	assert dataset.shape[0] == len(targets)

	evidence = dict(InteractNet=0)
	inf = inference(bn, evidence)
	inf = likelihood_from_inference(inf)
	# print inf

	accuracy = inference_accuracy(dataset)
	print accuracy
	
