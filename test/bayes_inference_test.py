import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'bayes/')
from create_struct import *
from load_data import *
from parse_theme import *
from project_data import *

import itertools

net_evidence = ['InteractNet', 'RecruitNetGroup', 'OtherKnowledge', 'Involve', 'LiveAlone', 'WarningLettersStatements', 'TargetTyp', 'HighValueCivilian', 'FurtherAttacks']

ill_evidence = ['AwareGriev', 'OtherKnowledge', 'Isolated', 'FurtherAttacks', 'WarningLettersStatements', 'LiveAlone', 'Stress', 'HighValueCivilian', 'Getaway', 'MentalIll']

ideo_evidence = ['ReligChangeInt', 'HighValueCivilian', 'RecruitNetGroup', 'Ideology', 'LiveAlone', 'TargetTyp', 'Propaganda', 'Legitimise', 'LocPubPriv', 'OtherKnowledge', 'IdeoChangeInt', 'AwareIdeo', 'Religion', 'FurtherAttacks']


def likelihood_from_inference(inference):
	inf = dict()
	inference = json.loads(inference)
	for key, vals in inference.items():
		inf[key] = max(vals.items(), key=lambda x: x[1])
	return inf	

def evidence_from_theme(theme):
	if theme == 'net':
		return net_evidence
	elif theme == 'ill':
		return ill_evidence
	elif theme == 'ideo':
		return ideo_evidence
	else:
		print 'ERROR'	

def inference_accuracy(dataset, nodes, features, inf):
	accuracy = 0
	for instance in dataset:
		accuracy += inference_accuracy_for_instance(nodes, features, inf, instance)
	return float(accuracy) / len(dataset)	

def inference_accuracy_for_instance(nodes, features, inference, instance):
	nrs = 0
	for key, (val, prob) in inference.iteritems():
		index = features.index(key)
		if instance[index] == int(val):
			nrs+=1

	accuracy = float(nrs)/len(nodes)
	# print 'Accuracy ' + str(accuracy)
	return accuracy

def create_combinations_evidence(possible_evidence):
	combinations_possible_evidence = []

	# single elements
	for x in possible_evidence:
		combinations_possible_evidence.append([x])

	for i in xrange(2,len(possible_evidence)+1):
		combinations = list(itertools.combinations(possible_evidence,i))
		combinations_possible_evidence = combinations_possible_evidence + (map(list,combinations))

	return combinations_possible_evidence		

def create_evidence_from_list(x):
	ev = dict()
	for elem in x:
		ev[elem] = 1
	return ev

def propagate_evidence(bn, possible_evidence, features):
	combinations_possible_evidence = create_combinations_evidence(possible_evidence)
	for x in combinations_possible_evidence:
		evidence = create_evidence_from_list(x)
		inf = inference(bn, evidence)
		inf = likelihood_from_inference(inf)
		# print inf

		accuracy = inference_accuracy(dataset, bn.V, features, inf)
		print accuracy

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

	# evidence = dict(InteractNet=1)
	# evidence = dict(MentalIll=1)
	# evidence = dict(Religion=1, Ideology=4)

	possible_evidence = evidence_from_theme(theme)
	propagate_evidence(bn, possible_evidence, features)
	
