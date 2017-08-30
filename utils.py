# David Bell's general utilties

import csv
import os
import functools

def map_to_list(func, l):
	return list(map(func, l))

def file_path(curr_file, *path_elements):
	direc = os.path.dirname(os.path.abspath(curr_file))
	return os.path.join(direc, *path_elements)

def open_file(curr_file, rel_path, protocol='r'):
	newline = '' if os.name == 'nt' else '\n'
	return open(file_path(curr_file, rel_path), protocol, newline=newline)

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

def filter_by_name_frags(name, name_frags):
	if len(name_frags) == 0:
		yield name
	working_name = name
	for i, frag in enumerate(name_frags):
		print(working_name)
		try:
			idx = working_name.index(frag)
			frag_end_idx = idx + len(frag)
			working_name = working_name[frag_end_idx:]
			if i == len(name_frags) - 1:
				yield name
		except ValueError:
			break

# filters list of strings, leaving on elements that contain name frags, in order
# NOTE: uses first instance of name fragment and no other instance
def filter_list_by_name_frags(l, name_frags):
	for name in l:
		for matching_name in filter_by_name_frags(name, name_frags):
			yield matching_name

def all_in_dir(curr_file, path_to_dir):
	path_to_dir = file_path(curr_file, path_to_dir)
	return os.listdir(path_to_dir)

def all_files_from_dir(curr_file, path_to_dir):
	dir_path = file_path(curr_file, path_to_dir)
	all_entries = all_in_dir(curr_file, path_to_dir)
	return [name for name in all_entries if not os.path.isdir(os.path.join(dir_path, name))]

def all_dirs_from_dir(curr_file, path_to_dir):
	dir_path = file_path(curr_file, path_to_dir)
	all_entries = all_in_dir(curr_file, path_to_dir)
	return [name for name in all_entries if os.path.isdir(os.path.join(dir_path, name))]

# finds all files in root direct 'path_to_dir' that contain name fragments in order
def all_files_with_name_frags(curr_file, path_to_dir, name_frags):
	all_files = all_files_from_dir(curr_file, path_to_dir)
	return [f for f in filter_list_by_name_frags(all_files, name_frags)]

# finds all directories in root direct 'path_to_dir' that contain name fragments in order
def all_dirs_with_name_frags(curr_file, path_to_dir, name_frags):
	all_dirs = all_dirs_from_dir(curr_file, path_to_dir)
	return [d for d in filter_list_by_name_frags(all_dirs, name_frags)]

def add_matrix(dest, mat):
	for i, l in enumerate(mat):
		for j, item in enumerate(l):
			dest[i][j] += item

def normalize_matrix(mat, norming_factor):
	for i, l in enumerate(mat):
		for j, item in enumerate(l):
			mat[i][j] /= norming_factor

def read_matrix_file(curr_file, path, name=None):
	path = os.path.join(path, name) if name is not None else path
	return read_csv(curr_file, path, preprocess=preprocess)

# assumes all files are csvs that can be read using 'read_matrix_file'
# if 'files' is an array of strings, filters list for names that contain name fragments
# then computes averages from each value in the matrix across all filtered files
# returns (averaged arrays, files that were filtered out)
# if 'files' is None, does the same operation, but 'files' becomes all files found in 'path_to_dir'
def average_files(curr_file, path_to_dir, name_frags=[], files=None, print_on=False):
	if files is None:
		files = all_files_with_name_frags(curr_file, path_to_dir, name_frags)
	else:
		files = [f for f in filter_list_by_name_frags(files, name_frags)]
	if len(files) < 1:
		raise 'There should be more than 0 files'
	if print_on:
		print('FILE: %s' % files[0])
	result = read_matrix_file(curr_file, path_to_dir, files[0])
	for i in range(1, len(files)):
		if print_on:
			print('FILE: %s' % files[i])
		add_matrix(result, read_matrix_file(curr_file, path_to_dir, files[i]))
	normalize_matrix(result, len(files))
	return (result, files)

def batch_average_files(curr_file, path_to_dir, name_frag_sets):
	search_pool = all_files_from_dir(curr_file, path_to_dir)
	for name_frags in name_frag_sets:
		(averaged, files_used) = average_files(curr_file, path_to_dir, name_frags, files=search_pool)
		yield averaged
		search_pool = [f for f in search_pool if f not in files_used]

def reduce_mult(l):
	return functools.reduce(lambda e1, e2: e1 * e2, l, 1)

# effectively the same as cartesian, but only provide the magnitude of each array
# given [1, 2, 1] generates
# [ [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
#	[0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
#	[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ] 
def create_points_for_domain(domain):
	ranges = map_to_list(lambda dim: [i for i in range(0, dim)], domain)
	return cartesian(ranges)

# multidimensional generalization of a cartesian proces
# given [2, 4, 6] and [2, 5, 8, 9] generates
# [[2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6], [2, 5, 8, 9, 2, 5, 8, 9, 2, 5, 8, 9]]
def cartesian(*arrs):
	domain = map_to_list(lambda a: len(a), arrs)
	coordinate_lists = []
	for i, dim in enumerate(domain):
		coords = []
		mult = 1
		if i != len(domain) - 1:
			mult = reduce_mult(domain[i+1:])
		for e in arrs[i]:
			coords += (mult * [e])
		repeat_factor = reduce_mult(domain[0:i])
		if repeat_factor > 0:
			coords *= repeat_factor
		coordinate_lists.append(coords)
	return coordinate_lists