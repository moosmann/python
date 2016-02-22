#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:55:09 2015

@author: jmoosmann
"""
from __future__ import division
from builtins import super
import numpy as np
import astra
import matplotlib
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import os
# from unittest import TestCase
matplotlib.use("qt4agg")
# matplotlib.use("gtkagg")
import matplotlib.pyplot as plt

# pylint: disable=invalid-name


def ainfo():
    """Print information about ASTRA object in memory."""
    astra.data2d.info()
    astra.data3d.info()
    astra.projector.info()
    astra.algorithm.info()
    astra.matrix.info()


class Phantom(object):
    """ Create phantom data, i.e. empty numpay array

    Parameters
    ----------

    shape : tuple of int
        Defines the numpy array

    >>> phan = Phantom().data
    >>> print(phan[4,4,:])
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

    """

    def __init__(self, shape=(10, 10, 10)):
        self._shape = shape
        # create emtpy array
        self._data = np.zeros(self._shape)

        # def __new__(cls):
        #     return self.phan

    @property
    def data(self):
        """Returns numpy.ndarray."""
        return self._data

    @property
    def shape(self):
        """Returns shape of numpy.ndarray."""
        return self._data.shape

    @property
    def size(self):
        """Returns size of numpy.ndarray."""
        return self._data.size


class Cuboid(Phantom):
    """ Cuboid phantom.

    Parameters
    ----------

    diameters : tuple of floats.
        Relative length in [0, 1] determining the size of the cuboidal phantom.

    >>> phan = Cuboid().data
    >>> print(phan[4, 4, :])
    [ 0.  0.  1.  1.  1.  1.  1.  1.  0.  0.]
    """

    def __init__(self, diameters=(0.2, 0.2, 0.2), **kwds):
        super(Cuboid, self).__init__(**kwds)
        self.diameters = diameters

        # create cuboid
        nl = np.ceil([a * (b - 1) for a, b in zip(self.diameters, self.shape)])
        nr = np.ceil(
            [(1 - a) * (b - 1) for a, b in zip(self.diameters, self.shape)])
        self._data[nl[0]:nr[0], nl[1]:nr[1], nl[2]:nr[2]] = 1


class HollowCuboid(Cuboid):
    """ Hollow cuboid phantom.

    >>> phan = HollowCuboid().data
    >>> print(phan[4, 4, :])
    [ 0.  0.  1.  1.  0.  0.  1.  1.  0.  0.]
    """

    def __init__(self, inner_diameters=(0.4, 0.4, 0.4), **kwds):
        super(HollowCuboid, self).__init__(**kwds)
        self.inner_diameters = inner_diameters

        # inner hollow cuboid
        nl = np.ceil(
            [a * (b - 1) for a, b in zip(self.inner_diameters, self.shape)])
        nr = np.ceil(
            [(1 - a) * (b - 1) for a, b in zip(
                self.inner_diameters, self.shape)])
        self.data[nl[0]:nr[0], nl[1]:nr[1], nl[2]:nr[2]] = 0


def phantom_ball(num_voxel=None, relative_origin=None, relative_radius=0.3):
    """Create a 3D binary phantom object 'phan' with num_voxel and
    consisting of a ball with radius relative_radius*min(num_voxel)
    and located at num_voxel.*relative_origin.

    Parameters
    ----------
    num_voxel : list of integers of length 1 or 3

        If len(list)==1, num_voxel is extended to length 3 with value
        num_voxel[0].

    relative_origin : list floats of length 1 or 3 with values in [0,1]

        If len(list)==1, num_voxel is extended to length 3 with value
        num_voxel[0].

    relative_radius : float

    Returns
    -------
    phantom : numpy.array with numpy.array.ndims = 3

    >>> phan = phantom_ball([10])
    >>> print phan.shape
    (10, 10, 10)
    """

    if isinstance(num_voxel, (int, float)):
        num_voxel = num_voxel,
    if isinstance(relative_origin, (int, float)):
        relative_origin = relative_origin,

    # Default arguments
    if not relative_origin:
        relative_origin = (0.5,) * 3
    if not num_voxel:
        num_voxel = (100,) * 3
    # Default arguments, continued
    if len(num_voxel) == 1:
        num_voxel = (num_voxel[0],) * 3
    if len(relative_origin) == 1:
        relative_origin = (relative_origin[0],) * 3

    # create grid
    xx = np.arange(num_voxel[0])
    yy = np.arange(num_voxel[1])
    zz = np.arange(num_voxel[2])
    # 3D array
    xx, yy, zz = np.meshgrid(yy, xx, zz)
    # centre
    x0 = relative_origin[0] * (num_voxel[0] - 1)
    y0 = relative_origin[1] * (num_voxel[1] - 1)
    z0 = relative_origin[2] * (num_voxel[2] - 1)
    # Radius
    r = relative_radius * np.min(num_voxel)
    # phantom
    r = (xx - x0) ** 2 + (yy - y0) ** 2 + (zz - z0) ** 2 <= r ** 2
    phantom = np.zeros(num_voxel)
    phantom[r] = 1

    return phantom


def show_slices(array3d=np.ones((10, 10, 10)), fig_name=None, figsize=None,
                plt_show=True):
    """Show three orthogonal slice through the centre of the 3D volume 'array3d'.

    Parameters
    ----------

    param array3d : numpy.array
        3D volume: numpy array.
    fig_name: None | string, optional, default: None
        Name of window figure.
    fig_size: (float,float), optional, default: None
        Size of figure in inches.
    plt_show: Bool, optional, default: True
        Toggle off to use plt.show() explicitly. E.g. in the case when a bunch
        of figures is created and plt.show() is used to display all of them.

    >>> fig = show_slices(np.ones((100,100,200)),'test', figsize=(10,2),\
    plt_show=False)
    >>> plt.show(block=False)
    >>> plt.pause(2)
    >>> plt.close(fig)
    >>> print fig
    Figure(800x160)
    """
    if not fig_name:
        fig = plt.figure(fig_name, figsize=figsize)
        # print fig_name
    else:
        fig = plt.figure(fig_name, figsize=figsize)

    # plt.ion()

    nx, ny, nz = array3d.shape
    cm = plt.cm.Greys
    vmin = array3d.min()
    vmax = array3d.max()
    # plt.get_cmap('Greys')

    ax1 = fig.add_subplot(1, 3, 1)
    num_slice = np.rint(nx / 2) - 1
    ax1.imshow(array3d[num_slice, :, :], vmin=vmin, vmax=vmax, cmap=cm,
               interpolation='none')
    ax1.set_title("yz-slice %u of %u" % (num_slice, nx))
    # ax1.set_ylabel('ax1 ylabel')

    ax2 = fig.add_subplot(1, 3, 2)
    num_slice = np.rint(ny / 2) - 1
    ax2.imshow(array3d[:, num_slice, :], vmin=vmin, vmax=vmax, cmap=cm,
               interpolation='none')
    ax2.set_title("xz-slice %u of %u" % (num_slice, ny))
    # ax2.set_ylabel('ax2 ylabel')

    ax3 = fig.add_subplot(1, 3, 3)
    num_slice = np.rint(nz / 2) - 1
    im = ax3.imshow(array3d[:, :, num_slice], vmin=vmin, vmax=vmax, cmap=cm,
                    interpolation='none')
    ax3.set_title("xy-slice %u of %u" % (num_slice, nz))
    # ax3.set_ylabel('ax3 ylable')

    # Add colorbar

    # Alternative 1:
    # plt.colorbar(im, use_gridspec=True)

    # Alternative 2:
    # fig.subplots_adjust(right=0.87)
    # cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # Alternative 3:
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", "10%", pad="10%")
    plt.colorbar(im, cax=cax)

    # plt.annotate('annotation', xy=(2, 1), xytext=(2, 1))

    plt.tight_layout()

    if plt_show:
        plt.show()

    return fig


def slicing(vol):
    """Animated slicing through volume data.

    >>> vol = phantom_ball((80, 90, 100), 0.5)
    >>> ani = slicing(vol)
    >>> plt.pause(3)
    >>> plt.close()
    """

    plt.switch_backend('qt4agg')

    fig = plt.figure('Animated slicing')
    cm = plt.get_cmap('Greys')

    global nn
    nn = 0
    plt.imshow(vol[:, nn, :], cmap=cm, interpolation='none')

    def updatefig():  # updatefig(*args):
        """Helper function returning the image instance to the corresponding
        iteration number."""
        global nn
        nn += 1
        nn = np.mod(nn, vol.shape[1])
        return plt.imshow(vol[:, nn, :], cmap=cm, interpolation='none'),

    # blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)

    plt.show(block=False)
    # plt.show()

    return ani


def prange(nparray):
    """Print min, max, and mean of input numpy array

    Parameters
    ----------
    nparray : numpy.array

    >>> a = np.arange(3)
    >>> prange(a)
    min: 0, max: 2, mean: 1
    >>> a = np.arange(3) - 1j * np.arange(3)
    >>> prange(a)
    Real: min: 0, max: 2, mean: 1
    Imag: min: -2, max: 0, mean: -1
    """

    def _prange(strnm, _nparray):
        """Helper function."""
        print("%smin: %g, max: %g, mean: %g" % (
            strnm, _nparray.min(), _nparray.max(), _nparray.mean()))

    if np.iscomplexobj(nparray):
        _prange("Real: ", nparray.real)
        _prange("Imag: ", nparray.imag)
    else:
        _prange("", nparray)


class DisplayIntermediates(object):
    """Utility class to display 3 central cuts orthgonal to the main axes and
    convergence measures.
    """

    def __init__(self, verbose=True,
                 vol=np.ones((10, 10, 10)),
                 cpd=np.arange(10),
                 l2_du=np.arange(10),
                 plt_pause=0.1,
                 switch_backend=None):

        self.verbose = verbose
        if verbose is not True:
            return
        vol_ndim = vol.ndim
        self.vol = vol
        self.cpd = cpd
        self.l2_du = l2_du
        self.plt_pause = plt_pause

        # Graphical backend
        if switch_backend is not None:
            plt.switch_backend(switch_backend)

        fig = plt.figure('Optimization: intermediate reconstructions and '
                         'convergence measures')
        fig.suptitle('Iteration: 0 of %u' % cpd.size)
        self.fig = fig
        self.cm = plt.cm.Greys

        # Create subplots
        if vol_ndim == 2:
            ax1 = fig.add_subplot(1, 2, 1)
            ax4 = fig.add_subplot(1, 2, 2)
        elif vol_ndim == 3:
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            self.ax2 = ax2
            self.ax3 = ax3
        self.ax1 = ax1
        self.ax4 = ax4

        # image(s)
        if vol_ndim == 2:
            im1 = self.ax1.imshow(vol, cmap=self.cm, interpolation='none')
            # Set colorbar and change ticks format
            _, self.cax1, self.scal_form, _ = self.add_colorbar(im1)
            ax1.set_title('array')
            ax1.axis('off')
        elif vol_ndim == 3:
            sli0, sli1, sli2 = np.round(np.array(vol.shape) / 2.0).astype(int)
            self.sli0 = sli0
            self.sli1 = sli1
            self.sli2 = sli2
            im1 = ax1.imshow(vol[sli0, :, :], cmap=self.cm,
                             interpolation='none')
            im2 = ax2.imshow(vol[:, sli1, :], cmap=self.cm,
                                  interpolation='none')
            im3 = ax3.imshow(vol[:, :, sli2], cmap=self.cm,
                                  interpolation='none')
            # Set colorbar and change ticks format
            _, self.cax1, self.scal_form, _ = self.add_colorbar(im1)
            _, self.cax2, _, _ = self.add_colorbar(im2)
            _, self.cax3, _, _ = self.add_colorbar(im3)
            # Set axes titles
            ax1.set_title('array[%u,:,:]' % sli0)
            ax2.set_title('array[:,%u,:]' % sli1)
            ax3.set_title('array[:,:,%u]' % sli2)
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')

        # line plots
        self.xx = 1+np.arange(cpd.size)
        self.l1, = ax4.plot(self.xx, cpd, 'r')
        ax4.set_title('convergence')
        ax4.set_ylabel('cPD')
        ax4.set_xlabel('iteration')
        ax4.yaxis.get_major_formatter().set_powerlimits((-2, 2))
        ax4t = self.ax4.twinx()
        self.ax4t = ax4t
        ax4t.set_ylabel('||du||_2')
        ax4t.yaxis.get_major_formatter().set_powerlimits((-2, 2))
        self.l2, = ax4t.plot(self.xx, l2_du, 'b')
        # Add legend
        plt.legend([self.l1, self.l2], ['cPD', 'du'], loc='upper right')

        # show
        plt.tight_layout()
        plt.pause(self.plt_pause)

        self._counter = 0

    def add_colorbar(self, image):
        """Add colorbar to subplot image.

        :type image: matplotlib.image.AxesImage
        :param image: axes.imshow  instance
        :return: cbar, cax, divider
        """

        # Get axes from image instance: matplotlib.axes._subplots.AxesSubplot
        ax = image.get_axes()
        # Create divider for existing axes instance
        divider = make_axes_locatable(ax)
        # Append axes to the right of ax3, with 20% width of ax3
        cax = divider.append_axes("right", size="5%", pad=0.04)
        # Change colorbar ticks format
        scalform = ticker.ScalarFormatter()
        scalform.set_scientific(True)
        scalform.set_powerlimits((-1, 1))
        # Create colorbar in the appended axes
        # cbar = plt.colorbar(image, cax=cax, ticks=ticker.MultipleLocator(0.2),
        #                     format="%.2f")
        cbar = plt.colorbar(image, cax=cax, format=scalform)

        return cbar, cax, scalform, divider

    def update(self):
        """Update images and plots.
        """
        if self.verbose is not True:
            return

        vol_ndim = self.vol.ndim

        # number of iterations
        self._counter += 1
        self.fig.suptitle('Iteration: %u of %u' % (self._counter,
                                                   self.cpd.size))
        # Plots
        self.l1.set_data(self.xx, self.cpd)
        self.l2.set_data(self.xx, self.l2_du)
        # Reset limits
        if self.cpd.min() < self.cpd.max():
            self.ax4.set_ylim([self.cpd.min(), self.cpd.max()])
        if self.l2_du.min() < self.l2_du.max():
            self.ax4t.set_ylim([self.l2_du.min(), self.l2_du.max()])

        # Images
        plt.hold(True)
        if vol_ndim == 2:
            # Update images
            im1 = self.ax1.imshow(self.vol, cmap=self.cm, interpolation='none')
            # Update colorbar
            plt.colorbar(im1, cax=self.cax1, format=self.scal_form)
        elif vol_ndim == 3:
            # Update images
            im1 = self.ax1.imshow(self.vol[self.sli0, :, :], cmap=self.cm,
                                  interpolation='none')
            im2 = self.ax2.imshow(self.vol[:, self.sli1, :], cmap=self.cm,
                                  interpolation='none')
            im3 = self.ax3.imshow(self.vol[:, :, self.sli2], cmap=self.cm,
                                  interpolation='none')
            # Update colorbar
            plt.colorbar(im1, cax=self.cax1, format=self.scal_form)
            plt.colorbar(im2, cax=self.cax2, format=self.scal_form)
            plt.colorbar(im3, cax=self.cax3, format=self.scal_form)

        # plt.tight_layout()
        plt.pause(self.plt_pause)

    def show(self):
        """Display show after final image and plot update in order to avoid
        window blocking and as result a force quit window. If it works
        depends on the graphical backend.
        """
        if self.verbose is not True:
            return
        # plt.show(block=False)
        plt.show()

    def close(self):
        """Close figure windows. Not working so far."""
        if self.verbose is not True:
            return
        plt.close(self.fig)


def partial(f, axis, dx=1.0):
    """Returns the component of the gradient of an N-dimensional array along
     the n-th dimension (axis).

    The partial derivative is computed using second order accurate central
    differences in the interior and first (forward or backwards) differences
    at the boundaries. The returned gradient hence has the same shape as the
    input array.

    Parameters
    ----------
    f : array_like
        An N-dimensional array containing samples of a scalar function.
    axis : int in {0, 1, ..., N}
        Dimension along the partial derivatives is taken.
    dx : scalar, optional
        Scalar specifying the sample distance along the dimension 'axis'.
        Default distance: 1.

    Returns
    -------
    partial : numpy.ndarray
        Array of the same shape as `f` giving the partial derivative of `f`
        with respect to the 'axis' dimension.
    """

    f = np.asanyarray(f)
    ndim = len(f.shape)  # number of dimensions

    # create slice objects --- initially all are [:, :, ..., :]
    # noinspection PyTypeChecker
    slice1 = [slice(None)] * ndim
    # noinspection PyTypeChecker
    slice2 = [slice(None)] * ndim
    # noinspection PyTypeChecker
    slice3 = [slice(None)] * ndim

    if f.shape[axis] < 2:
        raise ValueError(
            "Shape of array too small to calculate difference quotient, "
            "at least two elements are required.")

    # Numerical differentiation: 1st order edges, 2nd order interior
    out = np.empty_like(f)

    slice1[axis] = slice(1, -1)
    slice2[axis] = slice(2, None)
    slice3[axis] = slice(None, -2)
    # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
    out[slice1] = (f[slice2] - f[slice3]) / 2.0

    slice1[axis] = 0
    slice2[axis] = 1
    slice3[axis] = 0
    # 1D equivalent -- out[0] = (y[1] - y[0])
    out[slice1] = (f[slice2] - f[slice3])

    slice1[axis] = -1
    slice2[axis] = -1
    slice3[axis] = -2
    # 1D equivalent -- out[-1] = (y[-1] - y[-2])
    out[slice1] = (f[slice2] - f[slice3])

    # divide by step size
    out /= dx

    return out


def grayify_cmap(cmap):
    """Return a lumininance-correct grayscale version of any matplotlib
    colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)


def show_colormap(cmap):
    """Show luminance of color profile.

    >>> show_colormap('jet')
    >>> show_colormap('cubehelix')
    >>> show_colormap('bone')
    >>> plt.show()
    """

    im = np.outer(np.ones(10), np.arange(100))
    fig, ax = plt.subplots(2, figsize=(6, 1.5),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.1)
    ax[0].imshow(im, cmap=cmap)
    ax[1].imshow(im, cmap=grayify_cmap(cmap))


def set_surface_pixel_to_zero(vol, npixel):
    """Set surface pixels to zero.

    :param vol: 3D numpy array
    :param npixel: number of pixel layers to be set to zero
    :return: 3D numpy array

    >>> v = np.ones((4,4,4))
    >>> print(v[0, :, :])
    [[ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]]
    >>> set_surface_pixel_to_zero(v, 1)
    >>> print(v[0, :, :])
    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]]
    >>> print(v[1, :, :])
    [[ 0.  0.  0.  0.]
     [ 0.  1.  1.  0.]
     [ 0.  1.  1.  0.]
     [ 0.  0.  0.  0.]]
    """
    if npixel == 0:
        return
    nn = npixel
    vol[:nn, :, :] = 0
    vol[-nn:, :, :] = 0
    vol[:, :nn, :] = 0
    vol[:, -nn:, :] = 0
    vol[:, :, :nn] = 0
    vol[:, :, -nn:] = 0


def ndvolume(n, N, dtype=None):
    """
    >>> ndvolume(5, 1)
    array([0, 1, 2, 3, 4])
    >>> ndvolume(4, 2)
    array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 2, 4, 6],
           [0, 3, 6, 9]])
    >>> ndvolume(1, 3).shape
    (1, 1, 1)
    """

    s = [1]
    v = np.arange(n, dtype=dtype)
    for _ in range(N-1):
        s.insert(0, n)
        v = v * v.reshape(s)

    return v

#if __name__ == '__main__':
#    import doctest
#    doctest.testmod()

# class UnitTestCaseTest(TestCase):
#     def func(self):
#         self.fail()
