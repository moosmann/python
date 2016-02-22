"""
Unit tests for implementation of Chambolle Pock method within ODL
framework.
"""

import unittest
from pyastra import ODLProjectorOld as ProjectorOld
from pyastra import ODLProjector as Projector
from pyastra import Geometry, TomODLProjector, TomODLGeometry
import numpy as np
import time
from odl.space.cartesian import Rn
import ctdata
import matplotlib.pyplot as plt
import utils

class ODLProjectorOldTestCase(unittest.TestCase):
    """Test case for ASTRA projector in ODL."""

    def setUp(self):
        # Timing
        self.start_time = time.time()

        # Geometry
        self.geom = Geometry()

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print "%s: %.3f" % (self.id(), t)
        self.projector.clear_astra_memory()

    def test_odlprojector_instance(self):

        # Create cubic unit volume
        vol_rn = Rn(self.geom.vol_size)
        vol = np.ones(self.geom.vol_shape)
        vol_rn_vec = vol_rn.element(vol.ravel())

        # Create projections
        proj_rn = Rn(self.geom.proj_size)
        proj = np.ones(self.geom.proj_size)
        proj_rn_vec = proj_rn.element(proj.ravel())

        vol_norm_0 = vol_rn_vec.norm()
        self.assertEqual(vol_norm_0**2, np.sqrt(self.geom.vol_size)**2)
        proj_norm_0 = proj_rn_vec.norm()
        self.assertEqual(proj_norm_0**2, np.sqrt(self.geom.proj_size)**2)

        # ODLProjector instance
        self.projector = ProjectorOld(self.geom, vol_rn_vec, proj_rn_vec)
        self.projector.forward()
        proj_norm_1 = proj_rn_vec.norm()
        self.assertNotEqual(proj_norm_0, proj_norm_1)

        self.projector.backward()
        vol_norm_1 = vol_rn_vec.norm()
        self.assertNotEqual(vol_norm_0, vol_norm_1)

        self.projector.forward(vol_rn_vec)
        proj_norm_2 = proj_rn_vec.norm()
        self.assertNotEqual(proj_norm_1, proj_norm_2)
        self.projector.backward(proj_rn_vec)
        vol_norm_2 = vol_rn_vec.norm()
        self.assertNotEqual(vol_norm_2, vol_norm_1)

        print 'vol norms:', vol_norm_0, vol_norm_1, vol_norm_2
        print 'proj norms', proj_norm_0, proj_norm_1, proj_norm_2


class ODLProjectorTestCase(unittest.TestCase):
    """Test case for ASTRA projector in ODL."""

    def setUp(self):
        # Timing
        self.start_time = time.time()

        # Geometry
        self.geom = Geometry(6)

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print "%s: %.3f" % (self.id(), t)

    def test_odlprojector_instance(self):

        # Create cubic unit volume
        vol_rn = Rn(self.geom.vol_size)
        vol = np.ones(self.geom.vol_shape)
        vol_rn_vec = vol_rn.element(vol.ravel())

        # Create projections
        proj_rn = Rn(self.geom.proj_size)
        proj = np.ones(self.geom.proj_size)
        proj_rn_vec = proj_rn.element(proj.ravel())

        vol_norm_0 = vol_rn_vec.norm()
        self.assertEqual(vol_norm_0**2, np.sqrt(self.geom.vol_size)**2)
        proj_norm_0 = proj_rn_vec.norm()
        self.assertEqual(proj_norm_0**2, np.sqrt(self.geom.proj_size)**2)

        # ODLProjector instance
        projector = Projector(self.geom, vol_rn, proj_rn)
        proj_rn_vec = projector.forward(vol_rn_vec)
        proj_norm_1 = proj_rn_vec.norm()
        self.assertNotEqual(proj_norm_0, proj_norm_1)

        vol_rn_vec = projector.backward(proj_rn_vec)
        vol_norm_1 = vol_rn_vec.norm()
        self.assertNotEqual(vol_norm_0, vol_norm_1)

        proj_rn_vec = projector.forward(vol_rn_vec)
        proj_norm_2 = proj_rn_vec.norm()
        self.assertNotEqual(proj_norm_1, proj_norm_2)
        vol_rn_vec = projector.backward(proj_rn_vec)
        vol_norm_2 = vol_rn_vec.norm()
        self.assertNotEqual(vol_norm_2, vol_norm_1)

        projector.clear_astra_memory()

        print 'vol norms:', vol_norm_0, vol_norm_1, vol_norm_2
        print 'proj norms', proj_norm_0, proj_norm_1, proj_norm_2

    def test_ndim(self):

        vshape = (88, 77)
        vsize = np.prod(vshape)
        cols = 99
        angles = np.linspace(0, 2 * np.pi, 111, endpoint=False)
        psize = cols * np.size(angles)
        geom = Geometry(geometry_type='parallel', scale_factor=1,
                        volume_shape=vshape, det_col_count=cols,
                        det_row_count=1, angles=angles)

        print 'Vol size: ', vsize
        print 'Proj size:', psize
        print 'Voxel size:', self.geom.voxel_size

        vol_rn = Rn(vsize)
        proj_rn = Rn(psize)
        projector = Projector(geom, vol_rn, proj_rn)

        proj = projector.forward(vol_rn.element(1))
        p = proj.data.reshape(geom.proj_shape)
        print 'Proj at 0 degree: max = ', p[0, :].max()

        vol = projector.backward(proj_rn.element(1))
        print vol.data.max()

        projector.clear_astra_memory()

    def test_adjoint_scaling(self):

        vol_rn = Rn(self.geom.vol_size)
        vol_rn_ones = vol_rn.element(1)

        proj_rn = Rn(self.geom.proj_size)
        proj_rn_ones = proj_rn.element(1)

        projector = Projector(self.geom, vol_rn, proj_rn)

        proj1 = projector.forward(vol_rn_ones)
        vol1 = projector.backward(proj_rn_ones)

        n1 = proj1.inner(proj_rn_ones)
        n2 = vol_rn_ones.inner(vol1)

        print('<A x, y> = <x, Ad y> : {0} = {1}'.format(n1, n2))
        print('<A x, y> / <x, Ad y> - 1 = {0}'.format(n1/n2 - 1))

        proj = projector.forward(vol_rn_ones)
        vol = projector.backward(proj)

        alpha = proj.norm()**2 / vol_rn._inner(vol, vol_rn_ones)
        print alpha

        projector.clear_astra_memory()


class TomODLProjectorTestCase(unittest.TestCase):
    """Test case for ASTRA projector in ODL."""

    def setUp(self):
        # Timing
        self.start_time = time.time()

        # Geometry class
        self.geom = TomODLGeometry

        # Projector class
        self.projector = TomODLProjector

        self.matfile = '/home/jmoosmann/data/matlab/pyastra_recos'

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print "%s: %.3f" % (self.id(), t)

    def test_parallel(self):
        print('PARALLEL')

        data = ctdata.sets['parallel']
        data.load()
        print ' Data detector width:', data.detector_width_mm
        print ' Data projections shape:', data.projections.shape

        geom = self.geom(data, 2*(100,), 2*(100,))
        print ' Detector pixel width:', geom.det_col_spacing, geom.det_row_spacing

        projector = self.projector(geom)

        # Adjoint <Ax,y>=<x,Ad y> with x[:]=1 and y[:]=1
        rn_vol0 = Rn(geom.vol_size).element(1)
        rn_proj0 = Rn(geom.proj_size).element(1)
        rn_proj = projector.forward(rn_vol0)
        rn_vol = projector.backward(rn_proj0)
        l = rn_proj.inner(rn_proj0)
        r = rn_vol0.inner(rn_vol)
        print(' Adjoint with x[:]=1 and y[:]=1:')
        print('  <Ax,y> = <x,Ad y> : {0} = {1}'.format(l, r))
        print('  |<Ax,y> - <x,Ad y>| = {0}'.format(np.abs(l - r)))
        print('  <Ax,y> / <x,Ad y>  -1 = {0}'.format(l / r -1))

        # Back-project phantom data
        rn_proj = Rn(geom.proj_size).element(data.projections.ravel())
        rn_bp = projector.backward(rn_proj)

        # FBP
        rn_fbp = projector.fbp(rn_proj)
        rec = np.reshape(rn_fbp.data, geom.vol_shape)

        import scipy.io as sio
        sio.savemat(self.matfile, {'parallel_fbp': rec})
        # plt.imshow(rec, cmap=plt.cm.Greys)
        # plt.show()

        projector.clear_astra_memory()

    def test_fanflat(self):
        print('FANFLAT')

        data = ctdata.sets['fanflat']
        data.load()
        print 'Data detector width:', data.detector_width_mm
        print 'Data projections shape:', data.projections.shape

        wr = data.roi_cubic_width_mm
        print ' ROI width:', wr

        # geom = self.geom(data, 2*(100,), 2*(np.floor(wr),))
        geom = self.geom(data, 2*(100,), 2*(100,))
        print ' Detector pixel width:', geom.det_col_spacing, geom.det_row_spacing

        print ' Vol width:', geom.vol_width_mm
        print ' Voxel width:', geom.voxel_width

        projector = self.projector(geom)

        # Adjoint <Ax,y>=<x,Ad y> with x[:]=1 and y[:]=1
        rn_vol0 = Rn(geom.vol_size).element(1)
        rn_proj0 = Rn(geom.proj_size).element(1)
        rn_proj = projector.forward(rn_vol0)
        rn_vol = projector.backward(rn_proj0)
        l = rn_proj.inner(rn_proj0)
        r = rn_vol0.inner(rn_vol)
        print(' Adjoint with x[:]=1 and y[:]=1:')
        print('  <Ax,y> = <x,Ad y> : {0} = {1}'.format(l, r))
        print('  |<Ax,y> - <x,Ad y>| = {0}'.format(np.abs(l - r)))
        print('  <Ax,y> / <x,Ad y>  -1 = {0}'.format(l / r - 1))

        # Back-project phantom data
        rn_proj = Rn(geom.proj_size).element(data.projections.ravel())
        rn_bp = projector.backward(rn_proj)

        # FBP
        rn_fbp = projector.fbp(rn_proj)
        rec = np.reshape(rn_fbp.data, geom.vol_shape)

        # plt.imshow(rec, cmap=plt.cm.Greys)
        # plt.show()

        projector.clear_astra_memory()

    def test_parallel3d(self):
        print('PARALLEL 3D')

        data = ctdata.sets['parallel3d']
        data.load()
        print ' Data detector width:', data.detector_width_mm
        print ' Data projections shape:', data.projections.shape


        geom = self.geom(data, 3*(100,), 3*(100,))

        projector = self.projector(geom)

        # Adjoint <Ax,y>=<x,Ad y> with x[:]=1 and y[:]=1
        rn_vol0 = Rn(geom.vol_size).element(1)
        rn_proj0 = Rn(geom.proj_size).element(1)
        rn_proj = projector.forward(rn_vol0)
        rn_vol = projector.backward(rn_proj0)
        l = rn_proj.inner(rn_proj0)
        r = rn_vol0.inner(rn_vol)
        print(' Adjoint with x[:]=1 and y[:]=1:')
        print('  <Ax,y> = <x,Ad y> : {0} = {1}'.format(l, r))
        print('  |<Ax,y> - <x,Ad y>| = {0}'.format(np.abs(l - r)))
        print('  <Ax,y> / <x,Ad y>  -1 = {0}'.format(l / r - 1))

        # Back-project phantom data
        rn_proj = Rn(geom.proj_size).element(data.projections.ravel())
        rn_bp = projector.backward(rn_proj)

        projector.clear_astra_memory()

        # FBP
        # rn_fbp = projector.fbp(rn_proj)
        # rec = np.reshape(rn_fbp.data, geom.vol_shape)

        # utils.show_slices(rec)
        # utils.show_slices(data.projections)

    def test_cone(self):
        print('CONE')

        data = ctdata.sets['cone']
        data.load()
        print 'Data detector width:', data.detector_width_mm
        print 'Data projections shape:', data.projections.shape

        # wr = data.roi_cubic_width_mm
        # print ' ROI width:', wr
        # geom = self.geom(data, 3*(100,), 3*(np.floor(wr),))
        self.geom()
        geom = self.geom(data, 3*(50,), 3*(100,))
        print ' Detector pixel width:', geom.det_col_spacing, \
            geom.det_col_spacing
        print ' Vol width:', geom.vol_width_mm
        print ' Voxel width:', geom.voxel_width
        projector = self.projector(geom)

        # Adjoint <Ax,y>=<x,Ad y> with x[:]=1 and y[:]=1
        rn_vol0 = Rn(geom.vol_size).element(1)
        rn_proj0 = Rn(geom.proj_size).element(1)
        rn_proj = projector.forward(rn_vol0)
        rn_vol = projector.backward(rn_proj0)
        l = rn_proj.inner(rn_proj0)
        r = rn_vol0.inner(rn_vol)
        print(' Adjoint with x[:]=1 and y[:]=1:')
        print('  <Ax,y> = <x,Ad y> : {0} = {1}'.format(l, r))
        print('  |<Ax,y> - <x,Ad y>| = {0}'.format(np.abs(l - r)))
        print('  <Ax,y> / <x,Ad y>  -1 = {0}'.format(l / r - 1))

        # Back-project phantom data
        rn_proj = Rn(geom.proj_size).element(data.projections.ravel())
        rn_bp = projector.backward(rn_proj)

        # FBP
        rn_fbp = projector.fbp(rn_proj)
        rec = np.reshape(rn_fbp.data, geom.vol_shape)

        projector.clear_astra_memory()

        # utils.show_slices(data.projections, 'Projections', plt_show=False)
        # utils.show_slices(rec, 'FDK reco')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ODLProjectorTestCase)
    unittest.TextTestRunner(verbosity=0).run(suite)
