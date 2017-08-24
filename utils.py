# David Bell's general utilties

import csv
import os

def file_path(curr_file, *path_elements):
	direc = os.path.dirname(os.path.abspath(curr_file))
	return os.path.join(direc, *path_elements)

def open_file(curr_file, rel_path, protocol='r'):
	return open(file_path(curr_file, rel_path), protocol)

# default preprocess function for parsing a csv of floats
def preprocess(row):
	new_row = []
	for i, item in enumerate(row):
		try:
			new_row.append(float(item))
		except ValueError as e:
			pass
	return new_row

def read_csv(curr_file, rel_path, num_to_discard=0, delimiter=',', preprocess=None):
	data_file = open_file(curr_file, rel_path)
	parsed_csv = csv.reader(data_file, delimiter=delimiter)
	all_rows = []

	count = 0
	for row in parsed_csv:
		if count >= num_to_discard:
			if preprocess is not None:
				row = preprocess(row)
			all_rows.append(row)
		count += 1
	data_file.close()
	return all_rows

def write_csv(curr_file, rel_path, delimiter=',', to_dump=None):
	data_file = open_file(curr_file, rel_path, 'w')
	csv_writer = csv.writer(data_file, delimiter=delimiter)
	if to_dump is not None:
		for row in to_dump:
			csv_writer.writerow(row)
	data_file.close()

def filter_by_ext(l, ext):
	for i in l:
		if i.endswith(ext):
			yield i

def all_files_with_ext(path_to_dir, ext):
	path_to_dir = file_path(__file__, path_to_dir)
	all_files = os.listdir(path_to_dir)
	return [f for f in filter_by_ext(all_files, ext)]

def add_matrix(dest, mat):
	for i, l in enumerate(mat):
		for j, item in enumerate(l):
			dest[i][j] += item

def normalize_matrix(mat, norming_factor):
	for i, l in enumerate(mat):
		for j, item in enumerate(l):
			mat[i][j] /= norming_factor

def read_matrix_file(path, name):
	return read_csv(__file__, os.path.join(path, name), preprocess=preprocess)

def average_files(path_to_dir, ext):
	files = all_files_with_ext(path_to_dir, ext)
	if len(files) < 1:
		raise 'There should be more than 0 files'
	print('FILE: %s' % files[0])
	initial = read_matrix_file(path_to_dir, files[0])
	for i in range(1, len(files)):
		print('FILE: %s' % files[i])
		add_matrix(initial, read_matrix_file(path_to_dir, files[i]))
	normalize_matrix(initial, len(files))
	return initial
