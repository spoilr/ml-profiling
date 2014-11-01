#!/usr/bin/env python

import xlrd

workbook = xlrd.open_workbook('../../Downloads/ip/project data.xlsx')
worksheet = workbook.sheet_by_name('Sheet1')

num_rows = worksheet.nrows
num_cells = worksheet.ncols

cell_names = []
sample = []

def get_row(curr_row):
	data_row = []
	row = worksheet.row(curr_row)
	for curr_cell in range(4, num_cells):
		cell_value = worksheet.cell_value(curr_row, curr_cell)
		data_row.append(cell_value)
	return data_row

def get_data():
	for curr_row in range(0, num_rows):
		if curr_row < 1:
			cell_names = get_row(curr_row)
		else:
			sample.append(get_row(curr_row))

	for name in cell_names:
		print name
	

if __name__ == "__main__":
    get_data()


	
				
			