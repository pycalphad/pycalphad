"""
Register a ``'triangular'`` projection with matplotlib to plot diagrams on
triangular axes.

Users should not have to instantiate the TriangularAxes class directly.
Instead, the projection name can be passed as a keyword argument to
matplotlib.

>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> plt.gca(projection='triangular')
>>> plt.scatter(np.random.random(10), np.random.random(10))

"""

from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Affine2D, BboxTransformTo
from matplotlib.projections import register_projection
import matplotlib.spines as mspines
import matplotlib.axis as maxis

import numpy as np


class TriangularAxes(Axes):
    """
    A custom class for triangular projections.
    """

    name = 'triangular'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_aspect(1, adjustable='box', anchor='SW')
        self.cla()

    def _init_axis(self):
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)
        self._update_transScale()

    def cla(self):
        """
        Hard-code axes limits to be on [0, 1] for both axes.

        Warning: Limits not on [0, 1] may lead to clipping issues!
        """
        # Don't forget to call the base class
        super().cla()

        x_min = 0
        y_min = 0
        x_max = 1
        y_max = 1
        x_spacing = 0.1
        y_spacing = 0.1
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())
        self.xaxis.set_ticks_position('bottom')
        self.yaxis.set_ticks_position('left')
        super().set_xlim(x_min, x_max)
        super().set_ylim(y_min, y_max)
        self.xaxis.set_ticks(np.arange(x_min, x_max+x_spacing, x_spacing))
        self.yaxis.set_ticks(np.arange(y_min, y_max+y_spacing, y_spacing))

    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """
        # This code is based off of matplotlib's example for a custom Hammer
        # projection. See: https://matplotlib.org/gallery/misc/custom_projection.html#sphx-glr-gallery-misc-custom-projection-py

        # This function makes heavy use of the Transform classes in
        # ``lib/matplotlib/transforms.py.`` For more information, see
        # the inline documentation there.

        # Affine2D.from_values(a, b, c, d, e, f) constructs an affine
        # transformation matrix of
        #    a c e
        #    b d f
        #    0 0 1

        # A useful reference for the different coordinate systems can be found
        # in a table in the matplotlib transforms tutorial:
        # https://matplotlib.org/tutorials/advanced/transforms_tutorial.html#transformations-tutorial

        # The goal of this transformation is to get from the data space to axes
        # space. We perform an affine transformation on the y-axis, i.e.
        # transforming the y-axis from (0, 1) to (0.5, sqrt(3)/2).
        self.transAffine = Affine2D.from_values(1., 0, 0.5, np.sqrt(3)/2., 0, 0)
        # Affine transformation along the dependent axis
        self.transAffinedep = Affine2D.from_values(1., 0, -0.5, np.sqrt(3)/2., 0, 0)

        # This is the transformation from axes space to display space.
        self.transAxes = BboxTransformTo(self.bbox)

        # The data transformation is the application of the affine
        # transformation from data to axes space, then from axes to display
        # space. The '+' operator applies these in order.
        self.transData = self.transAffine + self.transAxes

        # The main data transformation is set up.  Now deal with gridlines and
        # tick labels. For these, we want the same trasnform as the, so we
        # apply transData directly.
        self._xaxis_transform = self.transData
        self._xaxis_text1_transform = self.transData
        self._xaxis_text2_transform = self.transData

        self._yaxis_transform = self.transData
        self._yaxis_text1_transform = self.transData
        self._yaxis_text2_transform = self.transData

    def get_xaxis_transform(self, which='grid'):
        assert which in ['tick1', 'tick2', 'grid']
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        return super().get_xaxis_text1_transform(pad)[0], 'top', 'center'

    def get_xaxis_text2_transform(self, pad):
        return super().get_xaxis_text2_transform(pad)[0], 'top', 'center'

    def get_yaxis_transform(self, which='grid'):
        assert which in ['tick1', 'tick2', 'grid']
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad):
        return super().get_yaxis_text1_transform(pad)[0], 'center', 'right'

    def get_yaxis_text2_transform(self, pad):
        return super().get_yaxis_text2_transform(pad)[0], 'center', 'left'

    def _gen_axes_spines(self):
        # The dependent axis (right hand side) spine should be set to complete
        # the triangle, i.e. the spine from (1, 0) to (1, 1) will be
        # transformed to (1, 0) to (0.5, sqrt(3)/2).
        dep_spine = mspines.Spine.linear_spine(self, 'right')
        dep_spine.set_transform(self.transAffinedep + self.transAxes)
        return {
            'left': mspines.Spine.linear_spine(self, 'left'),
            'bottom': mspines.Spine.linear_spine(self, 'bottom'),
            'right': dep_spine,
        }

    def _gen_axes_patch(self):
        """
        Override this method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.
        Any data and gridlines will be clipped to this shape.
        """
        return Polygon([[0, 0], [0.5, np.sqrt(3)/2], [1, 0]], closed=True)

    # Interactive panning and zooming is not supported with this projection,
    # so we override all of the following methods to disable it.
    def can_zoom(self):
        """
        Return True if this axes support the zoom box
        """
        return False

    def start_pan(self, x, y, button):
        pass

    def end_pan(self):
        pass

    def drag_pan(self, button, key, x, y):
        pass

    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        """
        Set the label for the y-axis. Default rotation=60 degrees.

        Parameters
        ----------
        ylabel : str
            The label text.

        labelpad : float, default: None
            Spacing in points from the axes bounding box including ticks
            and tick labels.

        loc : {'bottom', 'center', 'top'}, default: `yaxis.labellocation`
            The label position. This is a high-level alternative for passing
            parameters *y* and *horizontalalignment*.

        Other Parameters
        ----------------
        **kwargs : `.Text` properties
            `.Text` properties control the appearance of the label.

        See Also
        --------
        text : Documents the properties supported by `.Text`.
        """
        kwargs.setdefault('rotation', 60)
        return super().set_ylabel(ylabel, fontdict, labelpad, loc=loc, **kwargs)


# Now register the projection with matplotlib so the user can select it.
register_projection(TriangularAxes)
