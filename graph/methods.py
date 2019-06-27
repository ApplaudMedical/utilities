from matplotlib import pyplot as plt
import numpy as np
from .graph_manager import GraphManager
from textwrap import wrap
from ..utils import map_to_list

def wrap_names(names, n=12):
	return map_to_list(lambda name: '\n'.join(wrap(name, n)), names)

def color_generator(colors, multiplicity=1):
	c_idx = [0]
	if multiplicity < 1 or not isinstance(multiplicity, int):
		raise ValueError("'multiplicity' must be a int >=1")
	if multiplicity != 1:
		new_colors = []
		for c in colors:
			for j in range(multiplicity):
				new_colors.append(c)
		colors = new_colors
	def gen():
		col = colors[c_idx[0]]
		c_idx[0] = (c_idx[0] + 1) % len(colors)
		return col
	return gen

def gen_plot(fig=None, pos=111, scale=1.5):
	gr = 1.62
	if fig is None:
		fig = plt.figure(figsize=(scale * gr, scale))
	ax = fig.add_subplot(pos)

	# remove top and right lines on graph
	for orientation in ['top', 'right']:
		ax.spines[orientation].set_visible(False)

	# confine tick marks to bottom and left of plot
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	return (fig, ax, GraphManager(fig, ax))

def scatter(ax, x, xerr, y, yerr, err_bar_thickness=0.5, color='black', err_bar_color='black', label='', s=3, marker=None, facecolors=None, capsize=3):
	ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', color=err_bar_color, elinewidth=err_bar_thickness, capthick=err_bar_thickness, capsize=capsize)
	ax.scatter(x, y, color=color, label=label, s=s, marker=marker, facecolors=color if facecolors is None else facecolors)

def bar(ax, x, x_widths, y, yerr, err_bar_thickness=0.5, color='black', edge_color=None, err_bar_color='black', label='', capsize=3):
	edge_color = color if edge_color is None else edge_color
	ax.bar(x, y, width=x_widths, color=color, label=label, edgecolor=edge_color, linewidth=0.7)
	ax.errorbar(x, y, yerr=yerr, fmt='none', color=err_bar_color, capsize=capsize, elinewidth=err_bar_thickness, capthick=err_bar_thickness)




# tableau20 colors borrowed from http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
	r, g, b = tableau20[i]
	tableau20[i] = (r / 255., g / 255., b / 255.)
