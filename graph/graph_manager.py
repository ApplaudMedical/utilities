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
				self.ax.plot([coord] * len(points), points, '--', lw=0.5, color='black', alpha=0.3, zorder=-1)
			else:
				self.ax.plot(points, [coord] * len(points), '--', lw=0.5, color='black', alpha=0.3, zorder=-1)

__all__ = ['GraphManager']