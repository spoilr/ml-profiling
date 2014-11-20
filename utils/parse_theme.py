from load_data import *

def parse_theme(theme):
	spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)

	if theme == 'ill':
		dataset = data.extract_illness_examples()
		features = data.illness_features
	elif theme == 'net':
		dataset = data.extract_network_examples()
		features = data.network_features
	elif theme == 'ideo':
		dataset = data.extract_ideology_examples()
		features = data.ideology_features	
	else:
		print 'Error parsing theme'

	return [dataset, features]	