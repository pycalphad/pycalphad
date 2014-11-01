"""
Support for triangular (ternary) plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.grid_finder as grid_finder
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost, \
     ParasiteAxesAuxTrans
from axisartist.grid_helper import GridHelperTriangular


def triangular_axes(fig):
	# ternary projection
	tr = Affine2D.from_values(1., 0, 0.5, np.sqrt(3)/2., 0, 0)
	# negative ternary projection for dependent axis
	neg_tr = Affine2D.from_values(1., 0, -0.5, np.sqrt(3)/2., 0, 0)
	# identity transform
	identity_tr = Affine2D.from_values(1, 0, 0, 1, 0, 0)

	grid_helper = GridHelperTriangular(tr, 
				    extremes=(0,1,0,1), 
				    grid_type="independent")
	# use null_locator to kill extra horizontal gridlines from dependent axis
	null_locator = grid_finder.MaxNLocator(1)
	dep_grid_helper = GridHelperTriangular(neg_tr, 
					extremes=(0,1,0,1), 
					grid_type="dependent", 
					grid_locator2=null_locator)

	# Add independent axes with gridlines
	ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
	
	fig.add_subplot(ax1)

	ax1.axis[:].set_visible(False)
	ax1.axis["bottom"].set_visible(True)
	ax1.axis["left"].set_visible(True)

	# Add dependent axis with gridlines
	ax2 = ParasiteAxesAuxTrans(ax1, 
			    identity_tr, 
			    "equal", 
			    grid_helper=dep_grid_helper)
	ax2.axis["right"] = ax2.get_grid_helper().new_floating_axis(0,
							     1,
							     axes=ax1)
	ax2.axis["right"].toggle(ticklabels=False)
	ax1.parasites.append(ax2)
	ax1.grid(True)
	ax2.grid(True)
	ax1.plot([])
	ax1.set_aspect(1.)
	return ax1