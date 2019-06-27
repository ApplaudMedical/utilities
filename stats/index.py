import pandas as pd
import os
import numpy as np
from ..utils import map_to_list

def wilk_shapiro(samples):
	ws_path = os.path.join(os.path.dirname(__file__), 'wilk_shapiro_coefs.csv')
	ws_coefs = pd.read_csv(ws_path, header=None)
	n = len(samples)
	print(ws_coefs)
	coefs_for_n = map_to_list(float, ws_coefs.iloc[n-2][0].split(' '))

	if len(coefs_for_n) * 2 < len(samples):
		coefs_for_n.append(0)

	print(n)
	print(coefs_for_n)
	coefs = []
	
	ws = 0
	i = 0
	half = (len(samples) - 1) / 2
	for s in sorted(samples):
		prefix = 1
		if i > half:
			j = (half - i) % half
		else:
			j = i
			prefix = -1
		ws += (prefix * coefs_for_n[int(j)] * s)
		coefs.append(prefix * coefs_for_n[int(j)])
		i += 1
	print(coefs)

	return np.square(ws) / np.var(samples)
