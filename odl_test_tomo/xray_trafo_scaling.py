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

"""Test for X-ray transforms."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
import numpy as np
import pytest
import astra

# Internal
import odl
from odl.tomo.util.testutils import skip_if_no_astra_cuda

@pytest.skip
np.set_printoptions(precision=3)
# DiscreteLp volume / reconstruction space
# xx = 5
# vpts = 5
xx = 5.5
vpts = 11
# xx = 5.25
# vpts = 21
# xx = 8
# vpts = 16
# xx = 6
# vpts = 6
discr_vol_space_2d = odl.uniform_discr([-xx] * 2, [xx] * 2, [vpts] * 2,
                                       dtype='float32')
discr_vol_space_3d = odl.uniform_discr([-xx] * 3, [xx] * 3, [vpts] * 3,
                                       dtype='float32')

# Angle grid
agrid = odl.uniform_sampling(-.25 * np.pi, 0.75 * np.pi, 2)
astride = float(agrid.stride)
num_angle = agrid.size

# Detector grid
# dx = 11
# dpts = 11
dx = 10.5
dpts = 21
# dx = 5.5
# dpts = 11
# dx = 5.25
# dpts = 21
dgrid_2d = odl.uniform_sampling(-dx, dx, dpts)
dy = 1.0 * dx
dgrid_3d = odl.uniform_sampling([-dx, -dy], [dx, dy], [dpts] * 2)

# Distances
src_radius = 10000
det_radius = 10000
mag = (src_radius + det_radius) / src_radius
dfb_c = (src_radius - discr_vol_space_2d.grid.max()) * (
    (dgrid_2d.convex_hull().extent / 2 / (src_radius + det_radius)))
dfb_f = (src_radius + discr_vol_space_2d.grid.max()) * (
    (dgrid_2d.convex_hull().extent / 2 / (src_radius + det_radius)))

# Slice index to print
z_vol = np.round(discr_vol_space_3d.shape[2] / 2)
y_proj = np.floor(dgrid_3d.shape[1] / 2)

# Precision for adjoint test
precision = 2

# Print section
print('\n')
# print('angle: interval = {}, length = {}, extent = {}'
#       ''.format(angle_intvl, angle_intvl.length, angle_intvl.extent))
print('ANGLE: extent = {}, size = {}, stride = {}, grid = pi * {}'.format(
    agrid.extent(), agrid.size, agrid.stride,
    agrid.points().transpose() / np.pi))
print('magnification = ', mag)
print('dist of outermost beam from central beam: {} at entry, {} at exit'
      ''.format(dfb_c, dfb_f))
print('2D:')
print(' VOL: stride = {}, shape = {}, size = {}, min = {}, max = {}'.format(
    discr_vol_space_2d.grid.stride, discr_vol_space_2d.shape,
    discr_vol_space_2d.grid.size, discr_vol_space_2d.grid.min(),
    discr_vol_space_2d.grid.max()))
print(' VOL: grid = {}'.format(discr_vol_space_2d.points()[::vpts][:, 0]))
print(' DET grid: stride = {}, shape = {}'
      ''.format(dgrid_2d.stride, dgrid_2d.shape))
print(' DET: grid = {}'.format(dgrid_2d.points().transpose()))
print('3D:')
print(' VOL: stride = {}, shape = {}, size = {}, min = {}, max = {}'.format(
    discr_vol_space_3d.grid.stride, discr_vol_space_3d.shape,
    discr_vol_space_3d.grid.size, discr_vol_space_2d.grid.size,
    discr_vol_space_2d.grid.min(), discr_vol_space_2d.grid.max()))
print(' VOL: grid = {}'.format(discr_vol_space_3d.points()[:, 2][:vpts]))
print(' DET grid: stride = {}, shape = {}'
      ''.format(dgrid_3d.stride, dgrid_3d.shape))
print(' DET grid dim1 = {}'.format(dgrid_3d.points()[::dpts][:, 0]))
print(' DET grid dim2 = {}'.format(dgrid_3d.points()[:dpts][:, 1]))
print('\n vol z index = ', z_vol, '\n proj y index = ', y_proj)


@skip_if_no_astra_cuda
def test_xray_trafo_cpu_parallel2d():
    """2D parallel-beam discrete X-ray transform with ASTRA and CPU."""

    dgrid = dgrid_2d
    discr_vol_space = discr_vol_space_2d

    # Geometry
    geom = odl.tomo.Parallel2dGeometry(agrid, dgrid)

    # X-ray transform
    projector = odl.tomo.XrayTransform(discr_vol_space, geom,
                                       backend='astra_cpu', interp='linear')

    # Domain element
    f = projector.domain.one()
    f.norm()
    # Forward projection
    Af = projector(f)

    # Range element
    g = projector.range.one()

    # Back projection
    Adg = projector.adjoint(g)

    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    r = inner_vol / inner_proj

    # Adjoint matching
    # assert almost_equal(inner_vol, inner_proj, precision)

    print('\n\nCPU PARALLEL')
    # print('vol stride', projector.domain.grid.stride)
    # print('proj stride', projector.range.grid.stride)
    print('forward')
    print(Af.asarray())
    print('backward / angle_stride / num_angle')
    print(Adg.asarray() / astride / num_angle)
    print('sum(Af):', np.sum(Af.asarray()))
    print('<A f,g> =  ', inner_proj, '\n<f,Ad g> = ', inner_vol)
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))
    print('<Af,Af> = ', Af.inner(Af), '<f,Ad A f>', f.inner(
            projector.adjoint(Af)))


@skip_if_no_astra_cuda
def test_xray_trafo_cpu_fanflat():
    """2D fanbeam discrete X-ray transform with ASTRA and CUDA."""

    dgrid = dgrid_2d
    discr_vol_space = discr_vol_space_2d

    # Geometry
    geom = odl.tomo.FanFlatGeometry(agrid, dgrid, src_radius, det_radius)

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom, backend='astra_cpu',
                               interp='nearest')

    # Adjoint trafo
    Ad = odl.tomo.XrayBackProjector(discr_vol_space, geom, backend='astra_cpu')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    # Adg = A.adjoint(g)
    Adg = Ad(g)

    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    r = inner_vol / inner_proj

    # Adjoint matching
    # assert almost_equal(inner_vol, inner_proj, precision)

    print('\nCPU FANFLAT')
    # print('vol stride', A.domain.grid.stride)
    # print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[0])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray() / astride / num_angle)
    print('<A f,g> =  ', inner_proj, '\n<f,Ad g> = ', inner_vol)
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


@skip_if_no_astra_cuda
def test_xray_trafo_cuda_parallel2d():
    """2D parallel-beam discrete X-ray transform with ASTRA and CUDA."""

    dgrid = dgrid_2d
    discr_vol_space = discr_vol_space_2d

    # Geometry
    geom = odl.tomo.Parallel2dGeometry(agrid, dgrid)

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom, backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Adjoint matching
    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    # assert almost_equal(inner_vol, inner_proj, precision)

    print('\nCUDA PARALLEL 2D')
    # print('vol stride', A.domain.grid.stride)
    # print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[:])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray() / astride / num_angle)
    print('<A f,g> =  ', inner_proj, '\n<f,Ad g> = ', inner_vol)
    r = inner_vol / inner_proj
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


@skip_if_no_astra_cuda
def test_xray_trafo_cuda_fanflat():
    """2D fanbeam discrete X-ray transform with ASTRA and CUDA."""

    dgrid = dgrid_2d
    discr_vol_space = discr_vol_space_2d

    # Geometry
    geom = odl.tomo.FanFlatGeometry(agrid, dgrid, src_radius, det_radius)

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom, backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Adjoint matching
    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    # assert almost_equal(inner_vol, inner_proj, precision)

    print('\nCUDA FANFLAT')
    # print('vol stride', A.domain.grid.stride)
    # print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[0])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray() / astride / num_angle)
    print('<A f,g> =  ', inner_proj, '\n<f,Ad g> = ', inner_vol)
    r = inner_vol / inner_proj
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))
    print('<Af, Af> = {}'.format(Af.inner(Af)))
    print('<f, Ad Af> = {}'.format(f.inner(A.adjoint(Af))))


@skip_if_no_astra_cuda
def test_xray_trafo_cuda_parallel3d():
    """3D parallel-beam discrete X-ray transform with ASTRA CUDA."""

    dgrid = dgrid_3d
    discr_vol_space = discr_vol_space_3d

    # Geometry
    geom = odl.tomo.Parallel3dGeometry(agrid, dgrid)

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom, backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Adjoint matching
    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    # assert almost_equal(inner_vol, inner_proj, precision)
    vol_stride = A.domain.grid.stride
    proj_stride = A.range.grid.stride
    print('\nCUDA PARALLEL 3D')
    print('forward')
    print(Af.asarray()[0, :, y_proj])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray()[:, :, z_vol] / astride / num_angle)
    print('vol: stride = {}, shape = {}'.format(vol_stride, A.domain.shape))
    print('proj: stride = {}, shape = {}'.format(proj_stride, A.range.shape))
    print('<A f,g> = ', inner_proj, '\n<f,Ad g> =', inner_vol)
    print('<A f,g>/strides = {}\n<f,Ad g>/stride = {}'.format(
        inner_proj / np.prod(proj_stride),
        inner_vol / np.prod(vol_stride) / astride))
    r = inner_vol / inner_proj
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))

    print(np.sum((Adg.asarray() / astride) * f.asarray()))
    # r *= 1
    # print('scaled ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


@pytest.mark.skipif("not odl.tomo.ASTRA_CUDA_AVAILABLE")
def test_xray_trafo_cuda_conebeam_circular():
    """Cone-beam trafo with circular acquisition and ASTRA CUDA back-end."""

    dgrid = dgrid_3d

    # Geometry
    geom = odl.tomo.CircularConeFlatGeometry(agrid, dgrid, src_radius,
                                             det_radius, axis=[0, 0, 1])

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space_3d, geom, backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Adjoint matching
    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    # assert almost_equal(inner_vol, inner_proj, precision - 1)

    print('\nCUDA CONE CIRCULAR')
    # print('vol stride', A.domain.grid.stride)
    # print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[0, :, y_proj])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray()[:, :, z_vol] / astride / num_angle)
    print('<A f,g>: ', Af.inner(g), '\n<f,Ad g>', f.inner(Adg))
    r = inner_vol / inner_proj
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


@skip_if_no_astra_cuda
def test_xray_trafo_cuda_conebeam_helical():
    """Cone-beam trafo with helical acquisition and ASTRA CUDA back-end."""

    dgrid = dgrid_3d
    discr_vol_space = discr_vol_space_3d

    # Geometry
    geom = odl.tomo.HelicalConeFlatGeometry(agrid, dgrid, src_radius,
                                            det_radius, pitch=2,
                                            axis=[0, 0, 1])

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom, backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    assert geom.pitch != 0

    # Test adjoint matching
    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    # assert almost_equal(inner_proj, inner_vol, precision - 2)

    print('\nCUDA CONE HELICAL')
    # print('vol stride', A.domain.grid.stride)
    # print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[0, :, y_proj])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray()[:, :, z_vol] / astride / num_angle)
    print('<A f,g>: ', Af.inner(g), '\n<f,Ad g>', f.inner(Adg))
    r = inner_vol / inner_proj
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


def test_xray_trafo_cuda_parallel3d_direct():

    vol_shp = discr_vol_space_3d.grid.shape
    vol_min = discr_vol_space_3d.grid.min()
    vol_max = discr_vol_space_3d.grid.max()
    # given a 3D array of shape (x, y, z), a volume geometry is created as:
    #    astra.create_vol_geom(y, z, x, )
    # yielding a dictionary:
    #   'GridColCount': z
    #   'GridRowCount': y
    #   'GridSliceCount': x
    #   'WindowMinX': z_max
    #   'WindowMaxX': z_max
    #   'WindowMinY': y_min
    #   'WindowMaxY': y_min
    #   'WindowMinZ': x_min
    #   'WindowMaxZ': x_min
    vol_geom = astra.create_vol_geom(vol_shp[1], vol_shp[2], vol_shp[0],
                                     vol_min[2], vol_max[2],
                                     vol_min[1], vol_max[1],
                                     vol_min[0], vol_max[0])

    angles = agrid.points()
    detector_spacing_x = dgrid_3d.stride[1]
    detector_spacing_y = dgrid_3d.stride[0]
    det_row_count = dgrid_3d.shape[1]
    det_col_count = dgrid_3d.shape[0]
    proj_geom = astra.create_proj_geom(
        'parallel3d', detector_spacing_x, detector_spacing_y, det_row_count,
        det_col_count, angles)

    # Volume object
    vol_id = astra.data3d.create('-vol', vol_geom)

    # Projections object
    sino_id = astra.data3d.create('-sino', proj_geom)

    # Projector
    proj_cfg = {'type': 'cuda3d',
                'VolumeGeometry': vol_geom,
                'ProjectionGeometry': proj_geom}
    proj_id = astra.projector3d.create(proj_cfg)

    # Algorithm: FP
    alg_cfg_fp = {'type': 'FP3D_CUDA',
                  'ProjectorId': proj_id,
                  'ProjectionDataId': sino_id,
                  'VolumeDataId': vol_id}
    alg_id_fp = astra.algorithm.create(alg_cfg_fp)

    # Algorithm: BP
    alg_cfg_bp = {'type': 'BP3D_CUDA',
                  'ProjectorId': proj_id,
                  'ProjectionDataId': sino_id,
                  'ReconstructionDataId': vol_id}
    alg_id_bp = astra.algorithm.create(alg_cfg_bp)

    # Store volume data
    f = np.ones(vol_shp)
    astra.data3d.store(vol_id, f)

    # Forward projection
    astra.algorithm.run(alg_id_fp, 1)
    Af = astra.data3d.get(sino_id)

    # Store projection data
    g = np.ones((det_row_count, angles.size, det_col_count))
    astra.data3d.store(sino_id, g)

    # BP
    astra.algorithm.run(alg_id_bp, 1)
    Adg = astra.data3d.get(vol_id)

    # Strides
    astride = float(angles[1] - angles[0])
    vol_stride = (np.array(vol_max) - np.array(vol_min)) / np.array(vol_shp)
    proj_stride = (detector_spacing_x, detector_spacing_y, astride)

    print('\nDIRECT')
    print('<f,Adg> = {}\n<g, Af> = {}'.format(np.sum(f * Adg), np.sum(g * Af)))
    print('scaled:\n<f,Adg> = {}\n<g, Af> = {}'.format(
        np.sum(f * Adg) * np.prod(vol_stride) * astride / vol_stride[0],
        np.sum(g * Af) * np.prod(proj_stride)))


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
