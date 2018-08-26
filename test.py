from graph import *
import numpy as np

(fig, ax, gm) = gen_plot()

a = np.arange(0, 10, 10/1000)
for i in range(1, 5):
	ax.plot(a, np.power(a, i), tableau20[i])

ax.set_xlim(0)
ax.set_ylim(0)

gm.draw_refs()

fig.show()