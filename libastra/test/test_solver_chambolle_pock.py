"""
Test cases for ODL algorithms instances.
"""

from __future__ import print_function, division
# from __future__ import division
import unittest
import time
import numpy as np
import ctdata
from pyastra import ODLProjector, Geometry
from solver_chambolle_pock import ODLChambollePock
from odl.operator.solvers import Gradient, Divergence
import odl
from odl import Rn
from odl.diagnostics.operator import OperatorTest
# import matplotlib.pyplot as plt


class GradientOperatorTest(unittest.TestCase):

    def setUp(self):
        # Timing
        self.start_time = time.time()

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print("%s: %.3f" % (self.id(), t))

    def test_gradient_operator(self):
        # Volume geometry parameters
        # volGeom = Geometry(volume_shape=(2,3,4), voxel_size=(1,2,3))
        # vol_shape = (10, 11, 12)
        vol_shape = (20, 21, 22)
        v, v1, v2 = np.meshgrid(np.linspace(0, 1, vol_shape[1]),
                                np.linspace(0, 1, vol_shape[0]),
                                np.linspace(0, 1, vol_shape[2]))
        v = 1 + np.sin(np.pi*np.sqrt(v**2 + v1**2 + v2**2) + np.pi/2)
        v2 = 1 + np.sin(np.pi*np.sqrt(v**2 + v1**2 + v2**2) + np.pi/3)

        def set_surface_pixel_to_zero(vol, npixel):
            if npixel == 0:
                return
            nn = npixel
            vol[:nn, :, :] = 0
            vol[-nn:, :, :] = 0
            vol[:, :nn, :] = 0
            vol[:, -nn:, :] = 0
            vol[:, :, :nn] = 0
            vol[:, :, -nn:] = 0

        set_surface_pixel_to_zero(v, 0)
        set_surface_pixel_to_zero(v2, 0)

        vox_size = (1, 1, 1)
        ndim = len(vol_shape)
        volSize = [vol_shape[n] * vox_size[n] for n in range(ndim)]

        # Define continuous image space, typically a 2D or 3D volume
        volSpace = odl.L2(odl.Cuboid([0 for _ in range(ndim)], volSize))

        # Discretize the image space
        volDisc = odl.l2_uniform_discretization(volSpace, vol_shape)

        # Create data
        volVec = volDisc.element(v)

        # Target space: product space of image space
        volVec2 = volDisc.element(v2)
        gradDisc = odl.ProductSpace(volDisc, ndim)
        gradVec = gradDisc.element([volVec2, volVec2, volVec2])

        print('volDisc.norm:', volVec.norm()**2)
        print('gradDisc.norm:', gradVec.norm()**2)

        # Create gradient operator
        print('INITIALIZE GRADIENT OPERATOR:')
        grad = Gradient(volDisc, vox_size, edge_order=2, zero_padding=True)
        print(' op domain: ', grad.domain)
        print(' op range: ', grad.range)
        print(' op adjoint domain: ', grad.adjoint.domain)
        print(' op adjoint range: ', grad.adjoint.range)

        q = grad(volVec)
        p = grad.adjoint(gradVec)
        print('INNER PRODUCT:')
        print(' <Af,   g>', gradVec.inner(q))
        print(' < f, A*g>', volVec.inner(p))
        print(' Diff', gradVec.inner(q) -volVec.inner(p))
        s = 0
        v = volVec.asarray()
        for axis in range(ndim):
            slice1 = [slice(None)] * ndim
            slice2 = [slice(None)] * ndim
            slice1[axis] = 0
            slice2[axis] = -1

            vv = gradVec[axis].asarray()
            s1 = np.sum(v[slice1] * vv[slice1])
            s2 = np.sum(v[slice2] * vv[slice2])
            s += s1 - s2
            print(s1, s2, s1-s2, s)


        print( 'Surface contribution:', s)

        # print(type(gradDisc), type(gradDisc[0]))
        print('\nProduct space')
        print('gradDisc:', type(gradDisc))
        print('gradVec:', type(gradVec))
        print('q       :', type(q))
        print('gradDisc[0]:', type(gradDisc[0]))
        print('q[0]       :', type(q[0]))
        q2 = q.copy()

        # Create divergence operator
        div = Divergence(volDisc, vox_size, zero_padding=True)
        v = div(q)
        # print(type(v))

    def test_odl_operator_test(self):

        # Volume geometry parameters
        vol_shape = (20, 21, 22)
        vox_size = (1, 1, 1)
        ndim = len(vol_shape)
        volSize = [vol_shape[n] * vox_size[n] for n in range(ndim)]

        # Define continuous image space, typically a 2D or 3D volume
        volSpace = odl.L2(odl.Cuboid([0 for _ in range(ndim)], volSize))

        # Discretize the image space
        volDisc = odl.l2_uniform_discretization(volSpace, vol_shape)

        # Create data
        volVec = volDisc.element(1)

        # Create gradient operator
        grad = Gradient(volDisc, vox_size, edge_order=1, zero_padding=True)

        ot = OperatorTest(grad, operator_norm=1)
        ot.adjoint()

        # ot.adjoint()
        # ot.run_tests()


class ODLChambollePockTestCase(unittest.TestCase):
    """Test case for primal-dual method Chambolle-Pock algorithm for ODL"""

    def setUp(self):
        # Timing
        self.start_time = time.time()

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print("%s: %.3f" % (self.id(), t))

    def test_creation_of_vector_in_rn(self):

        geom = Geometry(2)

        rn = Rn(geom.proj_size)
        self.assertEqual(type(rn).__name__, 'Rn')
        rn_vec = rn.element(np.zeros(geom.proj_size))
        self.assertEqual(type(rn_vec).__name__, 'Vector')
        self.assertEqual(rn.dtype, 'float')
        self.assertEqual(rn.field, odl.RealNumbers())

        ODLChambollePock(geom)

    def test_adjoint_scaling_factor(self):
        """Test if back-projector A^* is adjoint of forward projector A:

            <A x, y>_D = <x,A^* y>_I .

        Define scaling factor as A^* = s B where is the implemented
        back-projector. Thus,

            s = <A x, y>_D / <x,B y>_I ,

        or using y = A x

            s = <A x, A x>_D / <x,B A x>_I .
        """
        geom = Geometry(2)

        # x = ones() and y = A x
        vol_rn = Rn(geom.vol_size)
        vol_rn_ones = vol_rn.element(1)
        proj_rn = Rn(geom.proj_size)
        projector = ODLProjector(geom, vol_rn, proj_rn)

        proj = projector.forward(vol_rn_ones)
        vol = projector.backward(proj)

        s0 = proj.norm() ** 2 / vol_rn._inner(vol, vol_rn_ones)

        # x = ones(), y = ones()
        vol_rn = Rn(geom.vol_size)
        vol_rn_ones = vol_rn.element(1)
        proj_rn = Rn(geom.proj_size)
        proj_rn_ones = proj_rn.element(1)

        projector = ODLProjector(geom, vol_rn, proj_rn)

        proj = projector.forward(vol_rn_ones)
        vol = projector.backward(proj_rn_ones)

        s1 = proj.inner(proj_rn_ones) / vol_rn_ones.inner(vol)

        # implemented function
        proj_vec = Rn(geom.proj_size).element(1)
        cp = ODLChambollePock(geom, proj_vec)
        cp.adjoint_scaling_factor()
        s2 = cp.adj_scal_fac
        self.assertEqual(s1, s2)

        print('Scaling factors:', s0, s1, s2)

        projector.clear_astra_memory()

    def test_matrix_norm(self):
        """Compute matrix norm of forward/backward projector using power
        norm. """

        geom = Geometry(2)
        proj_vec = Rn(geom.proj_size).element(1)

        # Compute norm for simple least squares
        cp = ODLChambollePock(geom, proj_vec)
        self.assertEqual(cp.adj_scal_fac, 1)
        mat_norm0 = cp.matrix_norm(iterations=4,
                                   vol_init=1,
                                   intermediate_results=True)
        self.assertTrue(mat_norm0[-1] > 0)

        # Resume computation
        mat_norm1, vol = cp.matrix_norm(iterations=3,
                                        vol_init=1, intermediate_results=True,
                                        return_volume=True)
        mat_norm2 = cp.matrix_norm(iterations=4, vol_init=vol,
                                   intermediate_results=True)
        self.assertNotEqual(mat_norm0[0], mat_norm2[0])

        self.assertEqual(mat_norm0[3], mat_norm2[0])

        # Compute norm for TV
        mat_norm3 = cp.matrix_norm(iterations=4, vol_init=1, tv_norm=True,
                                   intermediate_results=True)

        self.assertFalse(np.array_equal(mat_norm2, mat_norm3))
        print('LS unit init volume:', mat_norm2)
        print('TV unit init volume:', mat_norm3)

        # Use non-homogeneous initial volume
        v0 = np.random.rand(geom.vol_size)
        mat_norm4 = cp.matrix_norm(iterations=4, vol_init=v0, tv_norm=False,
                                   intermediate_results=True)
        mat_norm5 = cp.matrix_norm(iterations=4, vol_init=v0, tv_norm=True,
                                   intermediate_results=True)
        print('LS random init volume:', mat_norm4)
        print('TV random init volume:', mat_norm5)

        # test with adjoint scaling factor for backprojector
        self.assertEqual(cp.adj_scal_fac, 1)
        cp.adjoint_scaling_factor()
        self.assertFalse(cp.adj_scal_fac == 1)
        print('adjoint scaling factor:', cp.adj_scal_fac)

        mat_norm6 = cp.matrix_norm(iterations=4, vol_init=1, tv_norm=False,
                                   intermediate_results=True)
        mat_norm7 = cp.matrix_norm(iterations=4, vol_init=1, tv_norm=True,
                                   intermediate_results=True)

        print('LS init volume, adjoint rescaled:', mat_norm6)
        print('TV init volume, adjoint rescaled:', mat_norm7)

    def test_least_squares_method(self):
        geom = Geometry(2)
        proj_vec = Rn(geom.proj_size).element(1)
        cp = ODLChambollePock(geom, proj_vec)
        num_iter = 3
        cp.least_squares(num_iter, verbose=False)

    def test_tv(self):
        geom = Geometry(2)
        proj_vec = Rn(geom.proj_size).element(1)
        cp = ODLChambollePock(geom, proj_vec)
        cp.least_squares(3, L=131.0, non_negativiy_constraint=False,
                         tv_norm=1,
                         verbose=False)


class ODLChambollePockTestCaseGATE(unittest.TestCase):
    """Test case for primal-dual method Chambolle-Pock algorithm for ODL"""

    def setUp(self):
        # Timing
        self.start_time = time.time()

        # DATA
        d = ctdata.sets[14]
        # d.normalize = 10000
        d.load()
        det_row_count, num_proj, det_col_count = d.shape
        voxel_size_mm = 2 * d.roi_cubic_width_mm / det_col_count
        self.geom = Geometry(
            volume_shape=(det_col_count, det_col_count, det_row_count),
            det_row_count=det_row_count,
            det_col_count=det_col_count,
            angles=d.angles_rad,
            source_origin=d.distance_source_origin_mm / voxel_size_mm,
            origin_detector=d.distance_origin_detector_mm / voxel_size_mm,
            det_col_spacing=d.detector_width_mm/det_col_count/voxel_size_mm,
            det_row_spacing=d.detector_width_mm/det_row_count/voxel_size_mm,
            voxel_size=voxel_size_mm
        )
        self.voxel_size = voxel_size_mm

        # Rn vector
        self.proj_vec = Rn(self.geom.proj_size).element(
            d.projections.ravel() * (voxel_size_mm * 1e-3))

        # Class
        self.cp_class = ODLChambollePock
        # self.L = 271.47  # for data set 13 before projector rescaling
        self.L = 1.5  # TV for data set 13

        print ('Set up unit test')
        print ('  Data set:', d.filename)
        print ('  Raw data: min, max, mean = ', d.raw_data_min,
               d.raw_data_max, d.raw_data_mean)
        print('  g: min: %g, max: %g' % (self.proj_vec.data.min(),
                                         self.proj_vec.data.max()))
        print ('  Voxel size:', voxel_size_mm)
        print ('  Dector pixels:', self.geom.det_col_count,
               self.geom.det_row_count)
        print ('  Rel. pixel size:', self.geom.detector_spacing_x,
               self.geom.detector_spacing_x)
        # a = np.ones((10, 10))

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print ("%s: %.3f" % (self.id(), t))

    def test_adjoint_scaling_factor(self):

        # x
        vol_rn = Rn(self.geom.vol_size)
        vol_rn_ones = vol_rn.element(1)

        # y
        proj_rn = Rn(self.geom.proj_size)
        proj_rn_ones = proj_rn.element(1)

        # A
        projector = ODLProjector(self.geom, vol_rn, proj_rn)

        # A x
        proj = projector.forward(vol_rn_ones)
        # A^* y
        vol = projector.backward(proj_rn_ones)

        # scaling factor for x[:] = 1 and y[:] = 1
        s0 = proj.inner(proj_rn_ones) / vol_rn_ones.inner(vol)

        # A^* A x
        volp = projector.backward(proj)

        # scaling factor for x[:] = 1 and y = A x
        s1 = proj.norm() ** 2 / vol_rn._inner(volp, vol_rn_ones)

        cp = self.cp_class(self.geom, self.proj_vec)
        self.assertEqual(cp.adj_scal_fac, 1)

        cp.adjoint_scaling_factor()
        s2 = cp.adj_scal_fac
        self.assertFalse(s2 == 1)
        self.assertEqual(s0, s2)

        print ('Test adjoint')
        print ('  Scaling factor for backprojector', s0, s1, s2)

        projector.clear_astra_memory()

    def test_matrix_norm(self):
        """Compute matrix norm of forward/backward projector using power
        norm."""

        cp = self.cp_class(self.geom, self.proj_vec)
        cp.adjoint_scaling_factor()

        # Simple least squares
        mat_norm1 = cp.matrix_norm(iterations=20, vol_init=1,
                                   tv_norm=False,
                                   intermediate_results=True)

        # Least squares plus TV
        mat_norm2 = cp.matrix_norm(iterations=20, vol_init=1,
                                   tv_norm=True,
                                   intermediate_results=True)
        self.assertFalse(np.array_equal(mat_norm2, mat_norm1))
        print ('Test matrix norm')
        print ('  Adjoint scaling:', cp.adj_scal_fac)
        print ('  LS:', mat_norm1)
        print ('  LS + TV:', mat_norm2)
        print ('  Diff:', mat_norm2 - mat_norm1)

    def test_least_squares_method(self):
        # self.proj_vec.data[:] = self.proj_vec.data.max() -
        # self.proj_vec.data[:]
        # self.proj_vec[:] = (self.proj_vec.data[:] - self.proj_vec.data.min(
        # )) / (self.proj_vec.data.max() - self.proj_vec.data.min())
        print ('Proj: Min, Max = ', self.proj_vec.data.min(),
               self.proj_vec.data.max())
        cp = self.cp_class(self.geom, self.proj_vec)
        # cp.adjoint_scaling_factor()
        # print 'adjoint scaling', cp.adj_scal_fac
        cp.least_squares(iterations=20,
                         L=self.L,
                         non_negativiy_constraint=False,
                         verbose=True)

    def test_tv(self):
        cp = self.cp_class(self.geom, self.proj_vec)
        # cp.adjoint_scaling_factor()
        cp.least_squares(iterations=2,
                         L=self.L,
                         non_negativiy_constraint=True,
                         tv_norm=1,
                         verbose=False)


class ODLChambollePockTestCaseBox(unittest.TestCase):
    """Test case for primal-dual method Chambolle-Pock algorithm for ODL"""

    def setUp(self):
        # Timing
        self.start_time = time.time()

        # DATA
        d = ctdata.sets[15]
        # d.normalize = 10000
        d.load()
        det_row_count, num_proj, det_col_count = d.shape
        voxel_size_mm = 2 * d.roi_cubic_width_mm / det_col_count
        self.geom = Geometry(
            volume_shape=(det_col_count, det_col_count, det_row_count),
            det_row_count=det_row_count,
            det_col_count=det_col_count,
            angles=d.angles_rad,
            source_origin=d.distance_source_origin_mm / voxel_size_mm,
            origin_detector=d.distance_origin_detector_mm / voxel_size_mm,
            det_col_spacing=d.detector_width_mm/det_col_count/voxel_size_mm,
            det_row_spacing=d.detector_width_mm/det_row_count/voxel_size_mm,
            voxel_size=voxel_size_mm
        )
        self.voxel_size = voxel_size_mm

        # Rn vector
        self.proj_vec = Rn(self.geom.proj_size).element(
            d.projections.ravel() * (voxel_size_mm * 1e-3))

        # Class
        self.cp_class = ODLChambollePock
        # self.L = 271.47  # for data set 13 before projector rescaling
        self.L = 1.5  # TV for data set 13

        print('Set up unit test')
        print('  Data set:', d.filename)
        print('  Raw data: min, max, mean = ', d.raw_data_min,
               d.raw_data_max, d.raw_data_mean)
        print('  g: min: %g, max: %g' % (self.proj_vec.data.min(),
                                         self.proj_vec.data.max()))
        print('  Voxel size:', voxel_size_mm)

        # a = np.ones((10, 10))

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print("%s: %.3f" % (self.id(), t))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        ODLChambollePockTestCase)
    unittest.TextTestRunner(verbosity=0).run(suite)
    # unittest.main()
