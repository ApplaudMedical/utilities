import numpy as np
from scipy.optimize import curve_fit

def line(x, m, b):
	return m * x + b

def exp(x, a, b, c):
	return a * np.exp(b * x) + c

def fit(func, X, Y, p0=None, runtime=10000):
	params, cov = curve_fit(func, X, Y, p0=p0, maxfev=runtime)

	def parameterized_func(x):
		return func(x, *params)

	return (parameterized_func, params)

def r_squared(func, X, Y):
	return 1 - (np.sum(np.square(func(X) - Y)) / np.sum(np.square(np.mean(Y) - Y)))

__all__ = ['fit', 'line', 'exp', 'r_squared']