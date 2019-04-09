import numpy as np
from ..utils import select, to_unique_vals
from .methods import scatter

class SmartGraph:
	def __init__(self, fig, ax, data):
		self.fig = fig
		self.ax = ax
		self.data = data
		self.cache = {}

	def run_statistics(self, x_axis_col, y_axis_col):
		processed_data = []
		for group in self.data:
			unique_vals, = to_unique_vals(group, [x_axis_col])
			means = []
			cis = []
			for x_val in unique_vals:
				y_vals = select(group, {x_axis_col: x_val})[y_axis_col]
				means.append(y_vals.mean())
				cis.append(1.96 * y_vals.std() / np.sqrt(len(y_vals)))
			processed_data.append({'x_vals': unique_vals, 'y_vals': means, 'y_cis': cis})
		return processed_data

	def check_setting(self, setting, required_len, default):
		if setting is None:
			return [default] * required_len
		elif len(setting) != required_len:
			raise ValueError("Setting must be the 'required_len' %d" % required_len)
		else:
			return setting

	def plot(self, x_axis_col, y_axis_col, mode='scatter', err_bar_thickness=0.5, colors=None, err_bar_colors=None, labels=None, s=3, markers=None, facecolors=None):
		if mode in ['scatter', 'statistical']:
			self.ax.clear()
			colors = self.check_setting(colors, len(self.data), 'black')
			labels = self.check_setting(labels, len(self.data), '')
			markers = self.check_setting(markers, len(self.data), None)

			if mode == 'scatter':
				for i, group in enumerate(self.data):
					self.ax.scatter(group[x_axis_col], group[y_axis_col], c=colors[i], label=labels[i], marker=markers[i], s=s, facecolors=facecolors)
			else:
				err_bar_colors = self.check_setting(err_bar_colors, len(self.data), 'black')
				processed_data = self.run_statistics(x_axis_col, y_axis_col)
				for i, group in enumerate(processed_data):
					scatter(self.ax, group['x_vals'], None, group['y_vals'], group['y_cis'], err_bar_thickness=err_bar_thickness, color=colors[i], err_bar_color=err_bar_colors[i], label=labels[i], marker=markers[i], s=s, facecolors=facecolors)

	def toggle():
		pass
		#todo

__all__ = ['SmartGraph']