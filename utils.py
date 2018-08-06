	# David Bell's general utilties

import csv
import os
import functools
import numpy as np
import multiprocessing as mp

def map_to_list(func, l):
	return list(map(func, l))

def file_path(curr_file, *path_elements):
	direc = os.path.dirname(os.path.abspath(curr_file))
	return os.path.join(direc, *path_elements)

def open_file(curr_file, rel_path, protocol='r'):
	if protocol.endswith('b') or os.name != 'nt':
		return open(file_path(curr_file, rel_path), protocol)
	else:
		return open(file_path(curr_file, rel_path), protocol, newline='')

# default preprocess function for parsing a csv of floats
def preprocess(row):
	new_row = []
	for i, item in enumerate(row):
		try:
			new_row.append(float(item))
		except ValueError as e:
			pass
	return new_row

def read_csv(curr_file, rel_path, num_to_discard=0, delimiter=',', preprocess=None, sample_every=1):
	data_file = open_file(curr_file, rel_path)
	parsed_csv = csv.reader(data_file, delimiter=delimiter)
	all_rows = []

	for count, row in enumerate(parsed_csv):
		if count >= num_to_discard:
			if count % sample_every != 0:
				continue
			if preprocess is not None:
				row = preprocess(row)
				row = [row[i] for i in range(len(row)) if i % sample_every == 0]
			all_rows.append(row)
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

	expanded_name_frags = []
	for frag in name_frags:
		expanded_name_frags += frag.split('*')
	for i, frag in enumerate(expanded_name_frags):
		try:
			idx = working_name.index(frag)
			frag_end_idx = idx + len(frag)
			working_name = working_name[frag_end_idx:]
			if i == len(expanded_name_frags) - 1:
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
	if not isinstance(name_frags, list):
		name_frags = [name_frags]
	all_files = all_files_from_dir(curr_file, path_to_dir)
	return [f for f in filter_list_by_name_frags(all_files, name_frags)]

# finds all directories in root direct 'path_to_dir' that contain name fragments in order
def all_dirs_with_name_frags(curr_file, path_to_dir, name_frags):
	if not isinstance(name_frags, list):
		name_frags = [name_frags]
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

def read_matrix_file(curr_file, path, name=None, sample_every=1):
	path = os.path.join(path, name) if name is not None else path
	return read_csv(curr_file, path, preprocess=preprocess, sample_every=sample_every)

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
		raise ValueError('There should be more than 0 files')
	if print_on:
		print('FILE: %s' % files[0])
	result = read_matrix_file(curr_file, path_to_dir, files[0])
	for i in range(1, len(files)):
		if print_on:
			print('FILE: %s' % files[i])
		add_matrix(result, read_matrix_file(curr_file, path_to_dir, files[i]))
	normalize_matrix(result, len(files))
	return (result, files)

# untested
def max_over_files(curr_file, path_to_dir, name_frags=[], files=None, print_on=False):
	if files is None:
		files = all_files_with_name_frags(curr_file, path_to_dir, name_frags)
	else:
		files = [f for f in filter_list_by_name_frags(files, name_frags)]
	if len(files) < 1:
		raise ValueError('There should be more than 0 files')
	if print_on:
		print('FILE: %s' % files[0])
	maxs = read_matrix_file(curr_file, path_to_dir, files[0])
	print(maxs.shape)
	for i in range(1, len(files)):
		if print_on:
			print('FILE: %s' % files[i])
		maxs = np.array([maxs, read_matrix_file(curr_file, path_to_dir, files[i])]).max(0)
		print(maxs.shape)
	return (maxs, files)

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
# providing a 'true_dims' argument of same shape as 'points_per_axis' scales 'points_per_axis' to 'true_dims'
def create_points_for_domain(points_per_axis, true_dims=None):
	if true_dims is None:
		true_dims = points_per_axis
	ranges = []
	for i, dim in enumerate(points_per_axis):
		range_for_axis = [true_dims[i] * float(j) / dim for j in range(dim)]
		ranges.append(range_for_axis)
	return cartesian(*ranges)

# multidimensional generalization of a cartesian proces
# given [2, 4, 6] and [2, 5, 8, 9] generates
# [[2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6], [2, 5, 8, 9, 2, 5, 8, 9, 2, 5, 8, 9]]
def cartesian(*arrs):
	domain = map_to_list(lambda a: len(a), arrs)
	coordinate_lists = []
	print(arrs)
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

def mid_points(arr):
	mids = []
	for i in range(0, len(arr) - 1):
		mids.append((arr[i] + arr[i+1])/2)
	return mids

def pad_zeros(to_pad, length):
	padded = str(to_pad)
	while len(padded) < length:
		padded = '0' + padded
	return padded

# 2D only
def avg_with_mirror_along_axis(mat, axis):
	avgd = np.zeros(mat.shape)
	len_along_axis = mat.shape[axis]
	for i in range(mat.shape[axis]):
		if axis is 0:
			avgd[i] = (mat[i] + mat[len_along_axis - 1 - i]) / 2.
		else:
			avgd[:, i] = (mat[:, i] + mat[:, (len_along_axis - 1 - i)]) / 2.
	return avgd

def func_wrapper(args):
	func = args[0]
	args = args[1:]
	return func(*args)

def map_parallel(func, args_list, cores=None):
	cores = mp.cpu_count() if cores is None else cores
	args_list_with_func = map_to_list(lambda args: [func] + args, args_list)
	results = []

	for completed in range(0, len(args_list), cores):
		#print('Created pool')
		pool = mp.Pool(cores)
		partial_results = pool.map(func_wrapper, args_list_with_func[completed:(completed + cores)])
		pool.close()
		pool.join()
		results.append(partial_results)
		#print('Closed pool')
	results = [res for partial_results in results for res in partial_results]
	return results

def bucket(x, bucket_size):
	indices = np.linspace(x.min(), x.max(), int((x.max() - x.min()) / bucket_size) + 1)
	print(indices)
