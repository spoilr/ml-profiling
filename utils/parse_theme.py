from load_data import *
from project_data import *

def parse_theme(theme):
	spreadsheet = Spreadsheet(project_data_file)
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
	elif theme == 'all':
		dataset = data.extract_selected_examples()
		features = data.selected_features
	else:
		print 'Error parsing theme'

	return [dataset, features]	

def parse_theme_from_file(theme, file_name):
	spreadsheet = Spreadsheet(file_name)
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
	elif theme == 'all':
		dataset = data.extract_selected_examples()
		features = data.selected_features
	else:
		print 'Error parsing theme'

	return [dataset, features]		