''' Check only connected nodes as there are disconnected nodes too, but which can then be connected using expert knowledge'''


import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'bayes/')
from create_struct import *
from load_data import *
from parse_theme import *
from project_data import *
from combine_networks import *
from bayes_inference_test import inference_accuracy
from bayes_inference_test import inference_accuracy_for_instance
from bayes_inference_test import features_multiple_values
from bayes_inference_test import multiple_values
from bayes_inference_test import get_connected_nodes
from bayes_inference_test import create_evidence_from_list

from copy import deepcopy
import itertools

# net_evidence = ["LettersPost", "Financial", "BombManuals", "LiveAlone", "RecruitNetGroup", "InteractNet", "HighValueCivilian", "Discriminate", "FurtherAttacks", "OtherKnowledge", "NotCareInjustice", "Getaway", "Involve", "TargetTyp", "Interrupt", "Stockpile", "WarningLettersStatements", "NewMedia", "DryRuns"]
# ill_evidence = ["MentalIll", "Financial", "BombManuals", "LiveAlone", "LettersPost", "FurtherAttacks", "WarningLettersStatements", "NotCareInjustice", "Isolated", "HurtOthers", "OtherKnowledge", "AwareGriev", "NewMedia", "Stress", "HighValueCivilian", "Getaway", "DryRuns", "Interrupt"]
# ideo_evidence = ["Discriminate", "NotCareInjustice", "FurtherAttacks", "Religion", "AwareIdeo", "RelatStat", "IdeoChangeInt", "Interrupt", "Contradict", "Getaway", "Ideology", "LiveAlone", "TargetTyp", "Propaganda", "Financial", "LocPubPriv", "Stockpile", "RecruitNetGroup", "HighValueCivilian", "ReligChangeInt", "LettersPost", "Legitimise", "OtherKnowledge", "BombManuals"]

# net_evidence = ['BombManuals', 'LiveAlone', 'InteractNet', 'OtherKnowledge', 'FurtherAttacks', 'Involve', 'NewMedia', 'TargetTyp', 'DryRuns', 'RecruitNetGroup', 'Discriminate', 'Stockpile', 'WarningLettersStatements', 'HighValueCivilian']
# ill_evidence = ['LettersPost', 'Financial', 'FurtherAttacks', 'OtherKnowledge', 'Stress', 'Getaway', 'HurtOthers', 'DryRuns', 'AwareGriev', 'Interrupt', 'HighValueCivilian', 'NewMedia', 'WarningLettersStatements']
# ideo_evidence = ['LettersPost', 'BombManuals', 'Contradict', 'Legitimise', 'Ideology', 'RecruitNetGroup', 'Propaganda', 'Financial', 'LocPubPriv', 'LiveAlone', 'OtherKnowledge', 'ReligChangeInt', 'TargetTyp']

net_evidence = ['InteractNet', 'OtherKnowledge', 'Involve', 'NewMedia', 'RecruitNetGroup', 'WarningLettersStatements']

ill_evidence = ['Financial', 'OtherKnowledge', 'Stress', 'AwareGriev', 'NewMedia', 'WarningLettersStatements']

ideo_evidence = ['Legitimise', 'RecruitNetGroup', 'Propaganda', 'Financial', 'OtherKnowledge', 'ReligChangeInt']


def save_evidence(file_name, accuracy, evidence, nr_values):
	with open(file_name, "a") as myfile:	
		myfile.write('\n##############################')
	with open(file_name, "a") as myfile:	
		myfile.write('\nEvidence %s' % str(evidence))

	value = [(nr_values, accuracy)]	
	with open(file_name, "a") as myfile:	
		myfile.write('\nValue %s' % str(value))

def create_combinations_evidence(possible_evidence, start):
	combinations_possible_evidence = []
	for i in xrange(start,start+1):
		combinations = list(itertools.combinations(possible_evidence,i))
		combinations_possible_evidence = combinations_possible_evidence + (map(list,combinations))

	return combinations_possible_evidence		

def convert_values_to_string(dictionary):
	return {k: str(v) for k, v in dictionary.items()}

def likelihood_from_inference(inference):
	inf = dict()
	for key, vals in inference.items():
		inf[key] = max(vals.items(), key=lambda x: x[1])
	return inf	

def propagate_evidence(bn, possible_evidence, features, file_name, start, threhshold, nodes):
	combinations_possible_evidence = create_combinations_evidence(possible_evidence, start)
	for x in combinations_possible_evidence:
		# convert to string for join tree
		evidences = create_evidence_from_list(x)
		evidences = map(convert_values_to_string, evidences)

		for evidence in evidences:
			try:
				inf = jt_inference(bn, evidence)
				inf = likelihood_from_inference(inf)
				# print inf

				accuracy = inference_accuracy(dataset, nodes, features, inf, threhshold)
				save_evidence(file_name, accuracy, evidence, len(x))

				print accuracy
			except Exception:
				print 'Exception ' + str(evidence)

if __name__ == "__main__":
	bn_net = create_bayesian_network_structure('net')
	bn_ill = create_bayesian_network_structure('ill')
	bn_ideo = create_bayesian_network_structure('ideo')
	bn = combine_network(bn_net, bn_ill, bn_ideo)

	theme = 'all'
	spreadsheet = Spreadsheet(addendum_data_file, upsampling=False)
	data = Data(spreadsheet, upsampling=False)
	targets = np.array(data.targets)
	[dataset, features] = parse_theme_from_file(theme, addendum_data_file)

	dataset = np.hstack((dataset, targets.reshape(len(targets), 1))) # append targets
	features.append('HighValueCivilian')	# append target name in feature
	assert dataset.shape[0] == len(targets)

	nodes = get_connected_nodes(bn)
	print 'Connected nodes ' + str(nodes)
	possible_evidence = set(net_evidence + ill_evidence + ideo_evidence)
	start = int(raw_input("Start Combinations nr.\n"))
	threshold = float(raw_input("Threshold nr.\n"))
	file_name = theme + str(start) + "_evidence_feature_combined.txt"
	propagate_evidence(bn, possible_evidence, features, file_name, start, threshold, nodes)
	
