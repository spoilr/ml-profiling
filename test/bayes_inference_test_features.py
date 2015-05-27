''' Check only connected nodes as there are disconnected nodes too, but which can then be connected using expert knowledge'''


import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'bayes/')
from create_struct import *
from load_data import *
from parse_theme import *
from project_data import *
from bayes_inference_test import likelihood_from_inference
from bayes_inference_test import inference_accuracy
from bayes_inference_test import inference_accuracy_for_instance
from bayes_inference_test import features_multiple_values
from bayes_inference_test import multiple_values
from bayes_inference_test import get_connected_nodes
from bayes_inference_test import create_evidence_from_list

from copy import deepcopy
import itertools

net_evidence = ["LettersPost", "Financial", "BombManuals", "LiveAlone", "RecruitNetGroup", "InteractNet", "HighValueCivilian", "Discriminate", "FurtherAttacks", "OtherKnowledge", "NotCareInjustice", "Getaway", "Involve", "TargetTyp", "Interrupt", "Education", "Stockpile", "WarningLettersStatements", "NewMedia", "DryRuns"]

ill_evidence = ["MentalIll", "Financial", "BombManuals", "LiveAlone", "LettersPost", "FurtherAttacks", "WarningLettersStatements", "NotCareInjustice", "Isolated", "HurtOthers", "OtherKnowledge", "AwareGriev", "NewMedia", "Stress", "HighValueCivilian", "Getaway", "DryRuns", "Interrupt", "Education"]

ideo_evidence = ["Discriminate", "NotCareInjustice", "FurtherAttacks", "Religion", "AwareIdeo", "RelatStat", "IdeoChangeInt", "Interrupt", "Contradict", "Getaway", "Ideology", "LiveAlone", "TargetTyp", "Propaganda", "Financial", "LocPubPriv", "Stockpile", "RecruitNetGroup", "HighValueCivilian", "ReligChangeInt", "LettersPost", "Legitimise", "OtherKnowledge", "BombManuals", "Education"]

def evidence_from_theme(theme):
	if theme == 'net':
		return net_evidence
	elif theme == 'ill':
		return ill_evidence
	elif theme == 'ideo':
		return ideo_evidence
	else:
		print 'ERROR'	

def save_evidence(file_name, accuracy, evidence, nr_values):
	with open(file_name, "a") as myfile:	
		myfile.write('\n##############################')
	with open(file_name, "a") as myfile:	
		myfile.write('\nEvidence %s' % str(evidence))

	value = [(nr_values, accuracy)]	
	with open(file_name, "a") as myfile:	
		myfile.write('\nValue %s' % str(value))

def create_combinations_evidence(possible_evidence):
	combinations_possible_evidence = []

	# single elements
	for x in possible_evidence:
		combinations_possible_evidence.append([x])

	for i in xrange(2,len(possible_evidence)+1):
		combinations = list(itertools.combinations(possible_evidence,i))
		combinations_possible_evidence = combinations_possible_evidence + (map(list,combinations))

	return combinations_possible_evidence		

def propagate_evidence(theme, bn, possible_evidence, features, file_name):
	combinations_possible_evidence = create_combinations_evidence(possible_evidence)
	for x in combinations_possible_evidence:
		evidences = create_evidence_from_list(x)
		for evidence in evidences:
			try:
				inf = inference(bn, evidence)
				inf = likelihood_from_inference(inf)
				# print inf

				nodes = get_connected_nodes(bn)
				accuracy = inference_accuracy(dataset, nodes, features, inf)
				save_evidence(file_name, accuracy, evidence, len(x))

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

	possible_evidence = evidence_from_theme(theme)
	file_name = theme + "_evidence_feature.txt"
	propagate_evidence(theme, bn, possible_evidence, features, file_name)
	
