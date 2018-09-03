from matplotlib import pyplot as plt
import numpy as np

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

def gen_plot():
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# remove top and right lines on graph
	for orientation in ['top', 'right']:
		ax.spines[orientation].set_visible(False)

	# confine tick marks to bottom and left of plot
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	return (fig, ax, GraphManager(fig, ax))

def scatter(ax, x, xerr, y, yerr, err_bar_thickness=0.5, color='black', err_bar_color='black', label='', s=3):
	ax.scatter(x, y, color=color, label=label, s=s)
	ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', color=err_bar_color, capsize=3, elinewidth=err_bar_thickness, capthick=err_bar_thickness)

# tableau20 colors borrowed from http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
	r, g, b = tableau20[i]
	tableau20[i] = (r / 255., g / 255., b / 255.)

__all__ = ['gen_plot', 'scatter', 'tableau20', 'GraphManager']
