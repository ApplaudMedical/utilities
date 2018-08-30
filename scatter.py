from graph import *

def gen_scatter(x, xerr, y, yerr, err_bar_thickness=0.5, color='black', err_bar_color='black'):
	(fig, ax, gm) = gen_plot()

	ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', color=err_bar_color, capsize=3, elinewidth=err_bar_thickness, capthick=err_bar_thickness)
	ax.scatter(x, y, color=color)

	return (fig, ax, gm)

