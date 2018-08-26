from matplotlib import pyplot as plt

def gen_plot():
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# remove top and right lines on graph
	for orientation in ['top', 'right']:
		ax.spines[orientation].set_visible(False)

	# remove axis ticks
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	return (fig, ax)

# tableau20 colors borrowed from http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
	r, g, b = tableau20[i]
	tableau20[i] = (r / 255., g / 255., b / 255.)

__all__ = ['gen_plot', 'tableau20']
