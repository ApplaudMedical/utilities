import numpy as np
from ..utils import select, to_unique_vals, map_to_list
from .methods import scatter, bar
from functools import reduce

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
			stds = []
			unique_vals = [val for val in sorted(unique_vals)]
			for x_val in unique_vals:
				y_vals = select(group, {x_axis_col: x_val})[y_axis_col]
				means.append(y_vals.mean())
				cis.append(1.96 * y_vals.std() / np.sqrt(len(y_vals)))
				stds.append(y_vals.std())
			processed_data.append({'x_vals': unique_vals, 'y_vals': means, 'y_cis': cis, 'y_stds': stds})
		return processed_data

	def check_setting(self, setting, required_len, default):
		if setting is None:
			return [default] * required_len
		elif len(setting) != required_len:
			raise ValueError("Setting must be the 'required_len' %d" % required_len)
		else:
			return setting

	def set_bounds(self):
		x, y = (self.ax.get_xaxis(), self.ax.get_yaxis())
		self.ax.set_xlim(x.get_ticklocs()[0], x.get_ticklocs()[-1])
		self.ax.set_ylim(y.get_ticklocs()[0], y.get_ticklocs()[-1])

	def plot(self, x_axis_col, y_axis_col, mode='scatter', err_bar_thickness=0.5, colors=None, err_bar_colors=None, labels=None, s=3, markers=None, facecolors=None):
		if mode in ['scatter', 'statistical', 'bar']:
			self.ax.clear()
			colors = self.check_setting(colors, len(self.data), 'black')
			labels = self.check_setting(labels, len(self.data), '')
			markers = self.check_setting(markers, len(self.data), None)

			if mode == 'scatter':
				for i, group in enumerate(self.data):
					self.ax.scatter(group[x_axis_col], group[y_axis_col], c=colors[i], label=labels[i], marker=markers[i], s=s, facecolors=facecolors)
				self.set_bounds()
			elif mode == 'statistical':
				err_bar_colors = self.check_setting(err_bar_colors, len(self.data), 'black')
				processed_data = self.run_statistics(x_axis_col, y_axis_col)
				for i, group in enumerate(processed_data):
					scatter(self.ax, group['x_vals'], None, group['y_vals'], group['y_cis'], err_bar_thickness=err_bar_thickness, color=colors[i], err_bar_color=err_bar_colors[i], label=labels[i], marker=markers[i], s=s, facecolors=facecolors)
				x, y = (self.ax.get_xaxis(), self.ax.get_yaxis())
				self.set_bounds()
			else:
				x_group_padding = 0.2
				err_bar_colors = self.check_setting(err_bar_colors, len(self.data), 'black')
				processed_data = self.run_statistics(x_axis_col, y_axis_col)
				num_groups = len(processed_data)

				all_x_vals = map_to_list(lambda grp: set(grp['x_vals']), processed_data)
				all_x_vals = list([val for val in sorted(reduce(lambda x, y: x.union(y), all_x_vals))])

				for i, group in enumerate(processed_data):
					spaced_x_vals = []
					grp_x_val_idx = 0
					for j, x in enumerate(all_x_vals):
						if grp_x_val_idx < len(group['x_vals']) and x == group['x_vals'][grp_x_val_idx]:
							spaced_x_vals.append(j)
							grp_x_val_idx += 1

					x_vals_for_group = [((num_groups + x_group_padding) * k + i + 0.5) for k in spaced_x_vals]
					bar(self.ax, x_vals_for_group, 0.99, group['y_vals'], group['y_cis'], err_bar_thickness=err_bar_thickness, color=colors[i], err_bar_color='black', label=labels[i])
				self.ax.set_xlim(0, len(all_x_vals) * (num_groups) + (len(all_x_vals) - 1) * x_group_padding)
				self.ax.set_xticks([(float(num_groups) * (i + 0.5) + i * x_group_padding) for i in range(len(all_x_vals))])
				self.ax.set_xticklabels(all_x_vals)

	def toggle():
		pass
		#todo

__all__ = ['SmartGraph']