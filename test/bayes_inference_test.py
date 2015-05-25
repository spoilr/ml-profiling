''' Check only connected nodes as there are disconnected nodes too, but which can then be connected using expert knowledge'''


import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'bayes/')
from create_struct import *
from load_data import *
from parse_theme import *
from project_data import *

from copy import deepcopy
import itertools

# net_evidence = ['InteractNet', 'RecruitNetGroup', 'OtherKnowledge', 'Involve', 'LiveAlone', 'WarningLettersStatements', 'TargetTyp', 'HighValueCivilian', 'FurtherAttacks']

# ill_evidence = ['AwareGriev', 'OtherKnowledge', 'Isolated', 'FurtherAttacks', 'WarningLettersStatements', 'LiveAlone', 'Stress', 'HighValueCivilian', 'Getaway', 'MentalIll']

# ideo_evidence = ['ReligChangeInt', 'HighValueCivilian', 'RecruitNetGroup', 'Ideology', 'LiveAlone', 'TargetTyp', 'Propaganda', 'Legitimise', 'LocPubPriv', 'OtherKnowledge', 'IdeoChangeInt', 'AwareIdeo', 'Religion', 'FurtherAttacks']


net_evidence = ['InteractNet', 'RecruitNetGroup', 'OtherKnowledge']

ill_evidence = ['AwareGriev', 'OtherKnowledge', 'Isolated', 'Stress', 'MentalIll']

ideo_evidence = ['ReligChangeInt', 'RecruitNetGroup', 'Ideology', 'Propaganda', 'OtherKnowledge', 'IdeoChangeInt', 'AwareIdeo', 'Religion']


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
		if key in nodes:
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

features_multiple_values = ['HighValueCivilian', 'TargetTyp', 'LocPubPriv', 'ReligChangeInt', 'IdeoChangeInt']
multiple_values = dict(HighValueCivilian=[1,2], TargetTyp=[0,1,2], LocPubPriv=[0,1], IdeoChangeInt=[1,2,3,4], ReligChangeInt=[1,2,3,4])


def create_evidence_from_list(x):
	all_ev = []
	ev = dict()
	for elem in x:
		if elem not in multiple_values:
			if elem == 'Ideology':
				ev[elem] = 4
			elif elem == 'TargetTyp':
				ev[elem] = 0
			elif elem == 'LocPubPriv':
				ev[elem] = 0	
			elif elem == 'HighValueCivilian':
				ev[elem] = 2
			else:
				ev[elem] = 1

	if 'IdeoChangeInt' in x and 'ReligChangeInt' in x:
		for ideo, relig in itertools.product(multiple_values['IdeoChangeInt'],multiple_values['ReligChangeInt']):
			current_ev = deepcopy(ev)
			current_ev['IdeoChangeInt'] = ideo
			current_ev['ReligChangeInt'] = relig
			all_ev.append(current_ev)
	elif 'IdeoChangeInt' in x:
		for val in multiple_values['IdeoChangeInt']:
			current_ev = deepcopy(ev)
			current_ev['IdeoChangeInt'] = val
			all_ev.append(current_ev)
	elif 'ReligChangeInt' in x:	
		for val in multiple_values['IdeoChangeInt']:
			current_ev = deepcopy(ev)
			current_ev['ReligChangeInt'] = val
			all_ev.append(current_ev)
	else:
		all_ev.append(ev)
	return all_ev

def save_evidence(file_name, accuracy, evidence):
	with open(file_name, "a") as myfile:	
		myfile.write('\n##############################')
	with open(file_name, "a") as myfile:	
		myfile.write('\nEvidence %s' % str(evidence))
	with open(file_name, "a") as myfile:	
		myfile.write('\nAccuracy %f' % accuracy)

def get_connected_nodes(bn):
	connected_nodes = []
	vdata = bn.Vdata
	for node in vdata:
		if vdata[node]['parents'] or vdata[node]['children']:
			connected_nodes.append(node)
	return connected_nodes		

def propagate_evidence(bn, possible_evidence, features, file_name):
	combinations_possible_evidence = create_combinations_evidence(possible_evidence)
	for x in combinations_possible_evidence:
		evidences = create_evidence_from_list(x)
		for evidence in evidences:
			try:
				inf = inference(bn, evidence)
				inf = likelihood_from_inference(inf)
				# print inf

				nodes = get_connected_nodes(bn)
				print '\nConnected nodes ' + str(nodes)
				accuracy = inference_accuracy(dataset, nodes, features, inf)

				if accuracy >= 0.55:
					save_evidence(file_name, accuracy, evidence)

				print accuracy
			except Exception:
				print 'Exception ' + str(evidence)

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
	file_name = theme + "_evidence.txt"
	propagate_evidence(bn, possible_evidence, features, file_name)
	
