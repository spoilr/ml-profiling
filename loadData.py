#!/usr/bin/python

import xlrd
import themes

class Data:

	def __init__(self, spreadsheet):
		self.features = spreadsheet.features
		self.examples = spreadsheet.examples
		self.network_features = themes.network
		self.ideology_features = themes.ideology
		self.illness_features = themes.illness



class Spreadsheet:

	SHEET = 'Sheet1'
	FEATURE_ROW = 0
	START_EXAMPLE_ROW = 1

	def __init__(self, url):
		self.workbook = xlrd.open_workbook(url)
		self.worksheet = self.workbook.sheet_by_name(self.SHEET)
		self.features = self.get_cell_names()
		self.examples = self.get_examples()

	def get_cell_names(self):
		return self.get_row(self.FEATURE_ROW)

	def get_examples(self):
		num_rows = self.worksheet.nrows
		examples = []

		for curr_row in range(self.START_EXAMPLE_ROW, num_rows):
			examples.append(self.get_row(curr_row))

		return examples

	def get_row(self, curr_row):
		data_row = []
		row = self.worksheet.row(curr_row)
		num_cols = self.worksheet.ncols
		for curr_cell in range(0, num_cols):
			cell_value = self.worksheet.cell_value(curr_row, curr_cell)
			data_row.append(cell_value)
		return data_row

	

if __name__ == "__main__":
	spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)
