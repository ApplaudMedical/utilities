# David Bell's general utilties

import csv
import os
import functools
import numpy as np
import multiprocessing as mp
from scipy.interpolate import interp1d
import pandas as pd

def map_to_list(func, l):
	'''
	Maps the list 'l' through the function 'func'

	Parameters
	----------
	func : function
		Takes a single argument of type of 'l'
	l : list

	'''
	return list(map(func, l))

def file_path(curr_file, *path_elements):
	'''
	Joins absolute path to 'curr_file' with 'path_elements'

	Parameters
	----------
	curr_file : string
	*path_elements : strings
	'''
	direc = os.path.dirname(os.path.abspath(curr_file))
	return os.path.join(direc, *path_elements)

def open_file(curr_file, rel_path, protocol='r'):
	'''
	Opens file by joining 'curr_file', 'rel_path'.

	Parameters
	----------
	curr_file : string
	rel_path : string
	'''
	if protocol.endswith('b') or os.name != 'nt':
		return open(file_path(curr_file, rel_path), protocol)
	else:
		return open(file_path(curr_file, rel_path), protocol, newline='')

def make_dir_path(curr_file, dir_path):
	'''
	Makes a directory from the absolute path to 'curr_file' joined with 'dir_path'.
	https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output

	Parameters
	----------
	curr_file : string
	dir_path : string
	'''
	abs_path = file_path(curr_file, dir_path)
	if not os.path.exists(abs_path):
		try:
			os.makedirs(abs_path)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise

# default preprocess function for parsing a csv of floats
def preprocess(row):
	'''
	Converts list of strings in floats.

	Parameters
	----------
	row : list of strings
	'''
	new_row = []
	for i, item in enumerate(row):
		try:
			new_row.append(float(item))
		except ValueError as e:
			new_row.append(np.nan)
	return new_row

def read_csv(curr_file, rel_path, num_to_discard=0, delimiter=',', preprocess=None, sample_every=1, discarded=None):
	'''
	Reads csv specified by absolute path to 'curr_file' joined with 'rel_path'.

	Parameters
	----------
	curr_file : string
		Absolute path to 'curr_file' is used as the base path
	rel_path : string
		Joined to absolute path to 'curr_file' to specify location of CSV
	num_to_discard : int
		Number of rows to discard at beginning of CSV. These rows are placed into discarded.
	delimiter : string
		Delimiting character for parsing CSV
	preprocess : function
		Function to run on each row, which maps list --> list
	sample_every : int
		1 in every 'sample_every' number of rows are kept. Default is to save all rows.
	discarded : list
		Pass list to retrieve number of rows specified by 'num_to_discard'

	Returns
	-------
	Generator that returns processed rows
	'''
	data_file = open_file(curr_file, rel_path)
	parsed_csv = csv.reader(data_file, delimiter=delimiter)

	if num_to_discard > 0:
		for i in range(num_to_discard):
			row = parsed_csv.__next__()
			if discarded is not None and isinstance(discarded, list):
				discarded.append(row)

	def paginated_reader():
		for count, row in enumerate(parsed_csv):
			if count % sample_every != 0:
				continue
			if preprocess is not None:
				row = preprocess(row)
				row = [row[i] for i in range(len(row)) if i % sample_every == 0]
			yield row
		data_file.close()

	return paginated_reader()

def write_csv(curr_file, rel_path, delimiter=',', to_dump=None):
	'''
	Write to CSV specified by 'curr_file' and 'rel_path'

	Parameters
	----------
	curr_file : string
		Absolute path to 'curr_file' is used as the base path
	rel_path : string
		Joined to absolute path to 'curr_file' to specify location of CSV
	delimiter : string
		Delimiting character for writing CSV rows
	to_dump : list of lists or np array
		Data to write to file
	'''

	data_file = open_file(curr_file, rel_path, 'w')
	csv_writer = csv.writer(data_file, delimiter=delimiter)
	if to_dump is not None:
		for row in to_dump:
			csv_writer.writerow(row)
	data_file.close()

def filter_by_name_frags(name, name_frags, in_order=True):
	'''
	Yields 'name' if it separately contains all the strings within 'name_frags'. If 'in_order' is True, name fragements must be in order.

	Parameters
	----------
	name : string
		String to be returned if it contains all of 'name_frags'
	name_frags : list of strings
		Name fragments that 'name' must contain to be yielded
	in_order : boolean
		If true, 'name' must contain 'name_frags' in the order they are specified

	Returns
	-------
	Yields 'name' or nothing
	'''

	# simply yield name if there are no name fragments with which to filter
	if len(name_frags) == 0:
		yield name
	# working name is pruned as fragments are found within name to ensure fragments are found in order
	working_name = name

	# split any name fragments with '*' into two separate name fragments
	expanded_name_frags = []
	for frag in name_frags:
		expanded_name_frags += frag.split('*')

	for i, frag in enumerate(expanded_name_frags):
		try:
			idx = working_name.index(frag)
			frag_end_idx = idx + len(frag)
			if in_order:
				# if fragment is found in working name, prune working_name if order matters
				working_name = working_name[frag_end_idx:]
			if i == len(expanded_name_frags) - 1:
				# last fragment has been found; yield name
				yield name
		except ValueError:
			# name fragment does not exist within working_name; name is filtered out
			break

def filter_list_by_name_frags(l, name_frags, in_order=True):
	'''
	Filters strings contained in list 'l' by 'name_frags'. Each string within 'l' must contain
	the elements of 'name_frags' in order if 'in_order' is True or out of order if False

	Parameters
	----------
	l : list of strings
		List of strings to be filtered
	name_frags : list of strings
		Name fragments that each element of 'l' must contain to be yielded
	in_order : boolean
		If true, elements of 'l' must contain 'name_frags' in the order 'name_frags' specifies

	Returns
	-------
	Function itself is a generator that yields elements of 'l' that pass filtration
	'''
	for name in l:
		for matching_name in filter_by_name_frags(name, name_frags, in_order=in_order):
			yield matching_name

def all_in_dir(curr_file, path_to_dir):
	'''
	Returns the names of all files and directories within the directory specified by the absolute path to
	'curr_file' joined with 'path_to_dir'

	Parameters
	----------
	curr_file : string
		Absolute path to 'curr_file' is used as the base path
	path_to_dir : string
		Relative path from 'curr_file' to directory to read

	Returns
	-------
	List of strings of all files and directories in specified directory
	'''
	path_to_dir = file_path(curr_file, path_to_dir)
	return os.listdir(path_to_dir)

def all_files_from_dir(curr_file, path_to_dir):
	'''
	Similar to 'all_in_dir' but only returns files.

	Parameters
	----------
	curr_file : string
		Absolute path to 'curr_file' is used as the base path
	path_to_dir : string
		Relative path from 'curr_file' to directory to read
	'''
	dir_path = file_path(curr_file, path_to_dir)
	all_entries = all_in_dir(curr_file, path_to_dir)
	return [name for name in all_entries if not os.path.isdir(os.path.join(dir_path, name))]

def all_dirs_from_dir(curr_file, path_to_dir):
	'''
	Similar to 'all_in_dir' but only returns directories.

	Parameters
	----------
	curr_file : string
		Absolute path to 'curr_file' is used as the base path
	path_to_dir : string
		Relative path from 'curr_file' to directory to read
	'''
	dir_path = file_path(curr_file, path_to_dir)
	all_entries = all_in_dir(curr_file, path_to_dir)
	return [name for name in all_entries if os.path.isdir(os.path.join(dir_path, name))]

def all_files_with_name_frags(curr_file, path_to_dir, name_frags, in_order=True):
	'''
	Composes 'all_files_from_dir' and 'filter_list_by_name_frags' to read all files from a directory and filter them

	Parameters
	----------
	curr_file : string
		Absolute path to 'curr_file' is used as the base path
	path_to_dir : string
		Relative path from 'curr_file' to directory to read
	name_frags : list of strings
		Name fragments that each file in specified directory must contain to be yielded
	in_order : boolean
		If true, files in specified directory must contain 'name_frags' in the order 'name_frags' specifies

	Returns
	-------
	File names in specified directory that pass filtration
	'''
	if not isinstance(name_frags, list):
		name_frags = [name_frags]
	all_files = all_files_from_dir(curr_file, path_to_dir)
	return [f for f in filter_list_by_name_frags(all_files, name_frags, in_order=in_order)]

# finds all directories in root direct 'path_to_dir' that contain name fragments in order
def all_dirs_with_name_frags(curr_file, path_to_dir, name_frags, in_order=True):
	'''
	Composes 'all_dirs_from_dir' and 'filter_list_by_name_frags' to read all dirs from a directory and filter them

	Parameters
	----------
	curr_file : string
		Absolute path to 'curr_file' is used as the base path
	path_to_dir : string
		Relative path from 'curr_file' to directory to read
	name_frags : list of strings
		Name fragments that each dir in specified directory must contain to be yielded
	in_order : boolean
		If true, dirs in specified directory must contain 'name_frags' in the order 'name_frags' specifies

	Returns
	-------
	Dir names in specified directory that pass filtration
	'''
	if not isinstance(name_frags, list):
		name_frags = [name_frags]
	all_dirs = all_dirs_from_dir(curr_file, path_to_dir)
	return [d for d in filter_list_by_name_frags(all_dirs, name_frags, in_order=in_order)]

def add_matrix(dest, mat):
	'''
	Adds matrix 'mat' to matrix 'dest' (in-place). Matrices must have the same dimension, but can be lists of lists.

	Parameters
	----------
	dest : np.array or list of lists
		Destination matrix
	mat : np.array or list of lists
		Matrix to add to 'dest'

	Returns
	-------
	Nothing
	'''
	for i, l in enumerate(mat):
		for j, item in enumerate(l):
			dest[i][j] += item

def normalize_matrix(mat, norming_factor):
	'''
	Normalizes matrix 'mat' in-place by 'norming_factor'

	Parameters
	----------
	mat : np.array or list of lists
		Matrix to be normalized in-place
	norming_factor : float
		Normalization factor

	Returns
	-------
	Nothing
	'''
	for i, l in enumerate(mat):
		for j, item in enumerate(l):
			mat[i][j] /= norming_factor

def read_matrix_file(curr_file, path, name=None, sample_every=1, num_to_discard=0, discarded=None):
	path = os.path.join(path, name) if name is not None else path
	return read_csv(curr_file, path, preprocess=preprocess, sample_every=sample_every, num_to_discard=num_to_discard, discarded=discarded)

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

def diffs(arr):
	return [(arr[i+1] - arr[i]) for i in range(len(arr) - 1)]

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
		pool = mp.Pool(cores)
		partial_results = pool.map(func_wrapper, args_list_with_func[completed:(completed + cores)])
		pool.close()
		pool.join()
		results.append(partial_results)
	results = [res for partial_results in results for res in partial_results]
	return results

def t_test(group_1, group_2):
	group_1 = np.array(group_1)
	group_2 = np.array(group_2)
	mean_1 = np.mean(group_1)
	mean_2 = np.mean(group_2)
	std_1 = np.std(group_1)
	std_2 = np.std(group_2)
	return abs(mean_1 - mean_2) / np.sqrt(std_1**2 / len(group_1) + std_2**2 / len(group_2))

def confidence_interval(x, z=2.58):
	return z * np.std(x) / len(x)

# supports linear interpolation
def rebin(arr, new_dim):
	x = np.multiply(np.arange(len(arr)), float(new_dim + 1)/len(arr))
	iterp_func = interp1d(x, arr)
	return iterp_func(np.arange(new_dim))

# returns a list of unique values in the given Pandas dataframe for each column name specified 
def to_unique_vals(df, col_names):
	if type(col_names) is str:
		col_names = [col_names]
	return tuple([df[col_name].unique() for col_name in col_names])

# returns a subset of dataframe 
def select(df, selection):
    criteria = []
    for col in selection:
        criteria.append(df[col] == selection[col])
    return df[np.all(criteria, axis=0)]

def format_number(number, sig_figs=3):
    num = str(number)
    count = len(num.strip('.'))
    return num[:(-1 * (count - sig_figs - 1))]

def calc_bin_size(target_bin_size):
    growing = target_bin_size > 1
    return calc_bin_size_iter(target_bin_size, 1, growing)

def calc_bin_size_iter(target_bin_size, bin_size, growing=True):
    if bin_size > target_bin_size:
        if growing:
            return bin_size / 2
        else:
            return calc_bin_size_iter(target_bin_size, bin_size / 2)
    else:
        if not growing:
            return bin_size
        else:
            return calc_bin_size_iter(target_bin_size, 2 * bin_size)

def collapse_and_average(df, to_preserve, to_average):
    '''
    Returns a collapsed copy of the provided DataFrame 'df'. For each unique value in the column specified by
    to_preserve, values of the columns specified by 'to_average' are analyzed for mean, std, and count.
    
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame on which to run computations
    to_preserve : string or list of strings
        Names of columns to collapse to combinations of unique values
    to_average : list of strings
        Name of columns for which to compute mean, std, and count for each unique value of 'to_preserve'
    
    Returns
    -------
    Collapsed dataframe with statistics of 'to_average' added as additional columns
    '''
    try:
        to_unique_vals
        map_to_list
    except NameError as ne:
        print('Functions from utilities.utils are required.')
        raise ne

    to_preserve = [to_preserve] if type(to_preserve) is str else to_preserve

    rows = []
    unique_vals = to_unique_vals(df, to_preserve)
    num_to_preserve = len(unique_vals)
    all_combinations = cartesian(*unique_vals)

    for i, val in enumerate(unique_vals):
        rows.append([val])
        data_for_val = select(df, {to_preserve: val})
        for col in to_average:
            mean, std, count = data_for_val[col].mean(), data_for_val[col].std(), data_for_val[col].count()
            rows[i].append(mean)
            rows[i].append(std)
            rows[i].append(count)
            rows[i].append(std / np.sqrt(count) * 1.96)
    cols = [to_preserve]
    for element in map_to_list(lambda c: [c + ' AVG', c + ' STD', c + ' COUNT', c + ' CI'], to_average):
        for col in element:
            cols.append(col)
    
    return pd.DataFrame(data=rows, columns=cols)