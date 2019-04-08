from matplotlib import pyplot as plt
import numpy as np
from .utils import select, to_unique_vals

class GraphManager:
	def __init__(self, fig, ax):
		self.fig = fig
		self.ax = ax

	def draw_refs(self, horizontal=True, vertical=True):
		if horizontal:
			y = self.ax.get_yaxis()
			self.draw_ref_lines(y.get_ticklocs())
		if vertical:
			x = self.ax.get_xaxis()
			self.draw_ref_lines(x.get_ticklocs(), vertical=True)

	def draw_ref_lines(self, coords, vertical=False):
		if vertical:
			lims = self.ax.get_ylim()
		else:
			lims = self.ax.get_xlim()
		points = np.arange(lims[0], lims[1], np.abs(lims[1] - lims[0]) / 100)

		for coord in coords:
			if vertical:
				self.ax.plot([coord] * len(points), points, '--', lw=0.5, color='black', alpha=0.3)
			else:
				self.ax.plot(points, [coord] * len(points), '--', lw=0.5, color='black', alpha=0.3)

def color_generator(colors):
	c_idx = [0]
	def gen():
		col = colors[c_idx[0]]
		c_idx[0] = (c_idx[0] + 1) % len(colors)
		return col
	return gen

def gen_plot(fig=None, pos=111):
	if fig is None:
		fig = plt.figure()
	ax = fig.add_subplot(pos)

	# remove top and right lines on graph
	for orientation in ['top', 'right']:
		ax.spines[orientation].set_visible(False)

	# confine tick marks to bottom and left of plot
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	return (fig, ax, GraphManager(fig, ax))

def scatter(ax, x, xerr, y, yerr, err_bar_thickness=0.5, color='black', err_bar_color='black', label='', s=3, marker=None, facecolors=None):
	ax.scatter(x, y, color=color, label=label, s=s, marker=marker, facecolors=color if facecolors is None else facecolors)
	ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', color=err_bar_color, capsize=3, elinewidth=err_bar_thickness, capthick=err_bar_thickness)

def bar(ax, x, x_widths, y, yerr, err_bar_thickness=0.5, color='black', err_bar_color='black', label='')
	ax.errorbar(x, y, yerr=yerr, fmt='none', color=err_bar_color, capsize=3, elinewidth=err_bar_thickness, capthick=err_bar_thickness)
	ax.bar(x, y, width=x_widths, color=color, label=label)

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

# tableau20 colors borrowed from http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
	r, g, b = tableau20[i]
	tableau20[i] = (r / 255., g / 255., b / 255.)

__all__ = ['gen_plot', 'scatter', 'tableau20', 'GraphManager', 'color_generator', 'SmartGraph']
