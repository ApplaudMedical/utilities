import numpy as np
from scipy.optimize import curve_fit

def line(x, m, b):
	return np.multiply(m, x) + b

def line_zero_inter(x, m):
	return line(x, m, 0)

def exp(x, a, b, c):
	return a * np.exp(b * x) + c

def fit(func, X, Y, p0=None, runtime=10000, produce_fit=False):
	params, cov = curve_fit(func, X, Y, p0=p0, maxfev=runtime)

	def parameterized_func(x):
		return func(x, *params)

	if produce_fit:
		bounds = (X.min(), X.max())
		x_test = np.arange(bounds[0], bounds[1], (bounds[1] - bounds[0]) / 1000)
		return (parameterized_func, params, x_test, parameterized_func(x_test))
	return (parameterized_func, params)

def r_squared(func, X, Y):
	return 1 - (np.sum(np.square(func(X) - Y)) / np.sum(np.square(np.mean(Y) - Y)))

__all__ = ['fit', 'line', 'exp', 'r_squared']