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
from matplotlib.ticker import NullLocator, Formatter, FixedLocator
from matplotlib.transforms import Affine2D, BboxTransformTo, IdentityTransform
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

        # The main data transformation is set up.  Now deal with
        # gridlines and tick labels.

        # Longitude gridlines and ticklabels.  The input to these
        # transforms are in display space in x and axes space in y.
        # Therefore, the input values will be in range (-xmin, 0),
        # (xmax, 1).  The goal of these transforms is to go from that
        # space to display space.  The tick labels will be offset 4
        # pixels from the equator.

        self._xaxis_pretransform = IdentityTransform()
        self._xaxis_transform = self._xaxis_pretransform + self.transData
        self._xaxis_text1_transform = \
            Affine2D().scale(1.0, 0.0) + \
            self.transData + \
            Affine2D().translate(0.0, -20.0)
        self._xaxis_text2_transform = \
            Affine2D().scale(1.0, 0.0) + \
            self.transData + \
            Affine2D().translate(0.0, -4.0)

        # Now set up the transforms for the latitude ticks.  The input to
        # these transforms are in axes space in x and display space in
        # y.  Therefore, the input values will be in range (0, -ymin),
        # (1, ymax).  The goal of these transforms is to go from that
        # space to display space.  The tick labels will be offset 4
        # pixels from the edge of the axes ellipse.

        self._yaxis_transform = self.transData
        yaxis_text_base = self.transAffine + self.transAxes
        self._yaxis_text1_transform = yaxis_text_base + Affine2D().translate(-8.0, 0.0)
        self._yaxis_text2_transform = yaxis_text_base + Affine2D().translate(8.0, 0.0)

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


# Now register the projection with matplotlib so the user can select it.
register_projection(TriangularAxes)