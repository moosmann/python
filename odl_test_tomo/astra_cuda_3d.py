# Copyright 2014, 2015 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Example using the ASTRA CUDA for 3D geometries."""

# pylint: disable=invalid-name,no-name-in-module

from __future__ import print_function, division, absolute_import
import os.path as pth
from future import standard_library

standard_library.install_aliases()

# External
import numpy as np
import pytest
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Internal
import odl


@pytest.skip
def save_ortho_slices(data, name, sli):
    """Save three orthogonal slices of the 3D input volume.

    Parameters
    ----------
    data : `DiscreteLp`
    name : `str`
    sli : 3-element array-like
    """
    # indices for the orthogonal slices
    x, y, z = np.asarray(sli, int)
    path = pth.join(pth.dirname(pth.abspath(__file__)), 'astra_3d')

    data.show('imshow',
              saveto=pth.join(path, '{}_z{:03d}.png'.format(
                      name.replace(' ', '_'), z)),
              title='{} [:,:,{}]'.format(name, z),
              indices=[slice(None), slice(None), z])
    data.show('imshow',
              saveto=pth.join(path, '{}_y{:03d}.png'.format(
                      name.replace(' ', '_'), y)),
              title='{} [:,{},:]'.format(name, y),
              indices=[slice(None), y, slice(None)])
    data.show('imshow',
              saveto=pth.join(path, '{}_x{:03d}.png'.format(
                      name.replace(' ', '_'), x)),
              title='{} [{},:,:]'.format(name, x),
              indices=[x, slice(None), slice(None)])

    plt.close('all')

# `DiscreteLp` volume space
vol_shape = (80, 70, 60)
discr_vol_space = odl.uniform_discr([-40, -35, -30], [40, 35, 30],
                                    vol_shape, dtype='float32')
# Angles
angle_intvl = odl.Interval(0, 2 * np.pi)
angle_grid = odl.uniform_sampling(angle_intvl, 90, as_midp=False)

# Detector
dparams = odl.Rectangle([-50, -45], [50, 45])
det_grid = odl.uniform_sampling(dparams, (100, 90))

# Cone beam parameter
src_rad = 1000
det_rad = 10
pitch_factor = 0

# Create an element in the volume space
# discr_data = odl.util.phantom.cuboid(discr_vol_space,
#                                      (0.1, 0.15, 0.2,), (0.4, 0.35, 0.3))
discr_data = odl.util.phantom.indicate_proj_axis(discr_vol_space)
sli = 0.5
vol_cuts = np.round(sli * np.array(vol_shape))
save_ortho_slices(discr_data, 'phantom 3d cuda', vol_cuts)

# Geometry
axis = [0, 0, 1]
origin_to_det = [1, 0, 0]
geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, angle_grid,
                                   det_grid, axis=axis,
                                   origin_to_det=origin_to_det)
# Projection space
proj_space = odl.FunctionSpace(geom.params)

# `DiscreteLp` projection space
proj_shape = geom.grid.shape
discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                               dtype='float32')

# Indices of ortho slices
proj_cuts = (0, np.round(sli * proj_shape[1]),
             np.round(sli * proj_shape[2]))


def slicing(vol, axis=0):
    """Animated slicing through volume data.
    Parameters
    ----------
    vol : `ndarray`
    axis : positive `int`
    """

    if not np.isscalar(axis):
        axis = np.ndim(axis) - 1

    plt.switch_backend('qt4agg')
    fig = plt.figure('Animated slicing along axis {}'.format(axis))
    cm = plt.get_cmap('Greys')

    global nn
    nn = 0
    slo = [slice(None)] * 3
    slo[axis] = nn
    plt.imshow(vol[slo], cmap=cm, interpolation='none')

    def updatefig(*args):  # updatefig(*args):
        """Helper function returning the image instance to the corresponding
        iteration number."""
        global nn
        nn += 1
        nn = np.mod(nn, vol.shape[axis])
        slo[axis] = nn
        return plt.imshow(vol[slo], cmap=cm, interpolation='none'),

    frames = 100
    # blit=True: only re-draw the parts that have changed.
    ani = animation.FuncAnimation(fig, updatefig, frames=frames,
                                  interval=100, blit=True, repeat=False)

    plt.show(block=False)
    plt.show()

    return ani


def test_slicing():
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, angle_grid,
                                       det_grid, axis=axis,
                                       origin_to_det=origin_to_det)

    # Projections
    proj_data = odl.tomo.astra_cuda_forward_projector(discr_data, geom,
                                                      discr_proj_space)
    slicing(proj_data.asarray(), 0)


def test_proj():

    # `DiscreteLp` volume space
    vol_shape = (100,) * 3
    discr_vol_space = odl.uniform_discr([-50] * 3, [50] * 3, vol_shape,
                                        dtype='float32')

    # Angles: 0 and pi/2
    angle_intvl = odl.Interval(0, np.pi / 2)
    angle_grid = odl.uniform_sampling(angle_intvl, 2, as_midp=False)
    # agrid = angle_grid.points() / np.pi

    # Detector
    dparams = odl.Rectangle([-50] * 2, [50] * 2)
    det_grid = odl.uniform_sampling(dparams, (100, 100))

    def projector(index):

        axis = np.roll(np.array([1, 0, 0]), index)
        axis2 = np.roll(axis, 1)

        geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, angle_grid,
                                           det_grid, axis=axis,
                                           origin_to_det=axis2)
        # Projection space
        proj_space = odl.FunctionSpace(geom.params)
        discr_data = odl.util.phantom.indicate_proj_axis(discr_vol_space, 0.5)

        # `DiscreteLp` projection space
        proj_shape = geom.grid.shape
        discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                       dtype='float32')
        # Forward
        proj_data = odl.tomo.astra_cuda_forward_projector(
            discr_data, geom, discr_proj_space)
        return proj_data

    proj_data = projector(0)
    proj_data.show(indices=(0, np.s_[:], np.s_[:]))
    proj_data.show(indices=(1, np.s_[:], np.s_[:]))

    proj_data = projector(1)
    proj_data.show(indices=(0, np.s_[:], np.s_[:]))
    proj_data.show(indices=(1, np.s_[:], np.s_[:]))

    proj_data = projector(2)
    proj_data.show(indices=(0, np.s_[:], np.s_[:]))
    proj_data.show(indices=(1, np.s_[:], np.s_[:]))

    plt.show(block=True)


def test_yzproj():
    # `DiscreteLp` volume space
    vol_shape = (100,) * 3
    discr_vol_space = odl.uniform_discr([-50] * 3, [50] * 3, vol_shape,
                                        dtype='float32')

    # Angles: 0 and pi/2
    angle_intvl = odl.Interval(0, np.pi / 2)
    angle_grid = odl.uniform_sampling(angle_intvl, 2, as_midp=False)
    # agrid = angle_grid.points() / np.pi

    # Detector
    dparams = odl.Rectangle([-50] * 2, [50] * 2)
    det_grid = odl.uniform_sampling(dparams, (100, 100))

    axis = (0, 1, 0)
    origin_to_det = (0, 0, 1)
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams,
                                       angle_grid,
                                       det_grid, axis=axis,
                                       origin_to_det=origin_to_det)
    # Projection space
    proj_space = odl.FunctionSpace(geom.params)
    discr_data = odl.util.phantom.indicate_proj_axis(discr_vol_space, 0.5)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space,
                                                   proj_shape,
                                                   dtype='float32')
    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector(discr_data,
                                                      geom,

                                                      discr_proj_space)

    plt.switch_backend('qt4agg')
    proj_data.show(indices=np.s_[0, :, :])
    plt.show()


def test_astra_cuda_parallel3d():

    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, angle_grid,
                                       det_grid, axis=axis,
                                       origin_to_det=origin_to_det)

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector(discr_data, geom,
                                                      discr_proj_space)
    save_ortho_slices(proj_data, 'forward parallel 3d cuda', proj_cuts)

    # Backward
    rec_data = odl.tomo.astra_cuda_back_projector(proj_data, geom,
                                                  discr_vol_space)
    save_ortho_slices(rec_data, 'backward parallel 3d cuda', vol_cuts)


def test_astra_cuda_conebeam_circular():

    # Create geometries
    geom = odl.tomo.CircularConeFlatGeometry(
            angle_intvl, dparams, src_rad, det_rad, angle_grid, det_grid,
            axis=axis, src_to_det=origin_to_det)

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector(discr_data, geom,
                                                      discr_proj_space)

    save_ortho_slices(proj_data, 'forward conebeam circular cuda', proj_cuts)

    # Backward
    rec_data = odl.tomo.astra_cuda_back_projector(proj_data, geom,
                                                  discr_vol_space)

    save_ortho_slices(rec_data, 'backward conebeam circular cuda', vol_cuts)


def test_astra_cuda_conebeam_helical():

    # Create geometries
    geom = odl.tomo.HelicalConeFlatGeometry(
            angle_intvl, dparams, src_rad, det_rad, pitch_factor, angle_grid,
            det_grid, axis=axis, src_to_det=origin_to_det)

    save_ortho_slices(discr_data, 'phantom 3d cuda', vol_cuts)

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector(discr_data, geom,
                                                      discr_proj_space)

    save_ortho_slices(proj_data, 'forward conebeam helical cuda', proj_cuts)

    # Backward
    rec_data = odl.tomo.astra_cuda_back_projector(proj_data, geom,
                                                  discr_vol_space)

    save_ortho_slices(rec_data, 'backward conebeam helical cuda', vol_cuts)


def test_xray_trafo_parallel3d():
    """3D parallel-beam discrete X-ray transform with ASTRA CUDA."""

    # Discrete reconstruction space
    xx = 5
    nn = 4 * 5
    # xx = 5.5
    # nn = 11
    discr_vol_space3 = odl.uniform_discr([-xx] * 3, [xx] * 3, [nn] * 3,
                                         dtype='float32')

    # Angle
    angle_intvl = odl.Interval(0, 2 * np.pi) - np.pi / 4
    agrid = odl.uniform_sampling(angle_intvl, 4)

    # Detector
    # yy = 11
    # mm = 11
    yy = 10.5
    mm = 1*21
    dparams2 = odl.Rectangle([-yy, -yy], [yy, yy])
    dgrid2 = odl.uniform_sampling(dparams2, [mm] * 2)

    # Geometry
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams2, agrid, dgrid2)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space3, geom,
                               backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)
    A0f = odl.tomo.astra_cuda_forward_projector(f, geom,
                                                discr_proj_space)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)
    Adg0 = odl.tomo.astra_cuda_back_projector(g, geom,
                                              discr_vol_space3)

    print('\nvol stride', discr_vol_space3.grid.stride)
    print('proj stride', geom.grid.stride)
    print('angle intv:', angle_intvl.size)
    # f = discr_vol_space3.one()
    # print(f.asarray()[:, :, np.round(f.shape[2]/2)])
    print('forward')
    print(Af.asarray()[0, :, np.floor(Af.shape[2] / 2)])
    print(A0f.asarray()[0, :, np.floor(Af.shape[2] / 2)])
    print('backward')
    print(Adg.asarray()[:, :, np.round(f.shape[2] / 2)] / float(
            agrid.stride) / agrid.ntotal)
    print(Adg0.asarray()[:, :, np.round(f.shape[2] / 2)] / agrid.ntotal)


def test_xray_trafo_parallel2d():
    """3D parallel-beam discrete X-ray transform with ASTRA CUDA."""

    # Discrete reconstruction space
    xx = 5
    nn = 5
    # xx = 5.5
    # nn = 11
    discr_vol_space = odl.uniform_discr([-xx] * 2, [xx] * 2, [nn] * 2,
                                        dtype='float32')

    # Angle
    angle_intvl = odl.Interval(0, 2 * np.pi) - np.pi / 4
    agrid = odl.uniform_sampling(angle_intvl, 4)

    # Detector
    yy = 11
    mm = 11
    # yy = 10.5
    # mm = 21
    dparams = odl.Interval(-yy, yy)
    dgrid = odl.uniform_sampling(dparams, mm)

    # Geometry
    geom = odl.tomo.Parallel2dGeometry(angle_intvl, dparams, agrid, dgrid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom,
                               backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)
    A0f = odl.tomo.astra_cuda_forward_projector(f, geom,
                                                discr_proj_space)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)
    Adg0 = odl.tomo.astra_cuda_back_projector(g, geom,
                                              discr_vol_space)

    print('\nvol stride', discr_vol_space.grid.stride)
    print('proj stride', geom.grid.stride)
    print('angle intv:', angle_intvl.size)
    # f = discr_vol_space3.one()
    # print(f.asarray()[:, :, np.round(f.shape[2]/2)])
    print('forward')
    print(Af.asarray()[0])
    print(A0f.asarray()[0])
    print('backward')
    print(Adg.asarray() / float(agrid.stride) / agrid.ntotal)
    print(Adg0.asarray() / agrid.ntotal)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
