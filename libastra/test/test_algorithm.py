"""
Test cases for algorithms instances.
"""

import unittest
import time
import numpy as np
import ctdata
from pyastra import Projector
from algorithm import ChanGolubMullet, ChambollePock
# import matplotlib.pyplot as plt


class PartialDerivativeTest(unittest.TestCase):
    def test_partial_derivative(self):
        from utils import partial as pd
        a = np.array([[1, 1, 1, 1], [1, 2, 3, 4], [1, 4, 9, 16], [0, 0, 0, 0]],
                     dtype=np.float)
        a0 = pd(a, 0, 1.0)
        a1 = pd(a, 1)
        self.assertEqual(a.shape, a0.shape)
        self.assertEqual(a.shape, a1.shape)
        self.assertTrue(np.equal(a0[0, :], a[1, :] - 1).all())
        self.assertTrue(np.equal(a1[1, :], a[0, :]).all())


class TestGUI(unittest.TestCase):
    def test_gui(self):
        num_iter = 8
        u, p, cpd, l2_atp = ChambollePock(
            projections=np.ones((100, 180, 100)),
            projector=Projector()).least_squares(
            num_iterations=num_iter,
            L=363.569641113,
            verbose=True,
            non_negativiy_constraint=True)
        self.assertEqual(u.__class__.__name__, 'ndarray')
        self.assertEqual(cpd.size, num_iter)
        self.assertEqual(l2_atp.size, num_iter)


class ChanGolubMulletTestCase(unittest.TestCase):

    def setUp(self):
        # Timing
        self.start_time = time.time()

        # DATA
        self.d = ctdata.sets[14]
        self.d.load()

        # Parameters
        det_row_count, num_proj, det_col_count = self.d.shape
        num_voxel = (det_col_count, det_col_count, det_row_count)
        # voxel_size = 1
        voxel_size = 2 * self.d.roi_cubic_width_mm / num_voxel[0]
        source_origin = self.d.distance_source_origin_mm / voxel_size
        origin_detector = self.d.distance_origin_detector_mm / voxel_size
        angles = self.d.angles_rad
        det_col_spacing = self.d.detector_width_mm / det_col_count / voxel_size
        det_row_spacing = det_col_spacing

        # PROJECTOR
        self.projector = Projector(
            num_voxel=num_voxel,
            det_row_count=det_row_count, det_col_count=det_col_count,
            source_origin=source_origin, origin_detector=origin_detector,
            det_row_spacing=det_row_spacing, det_col_spacing=det_col_spacing,
            angles=angles)

        # ALGORITHM
        self.cgm = ChanGolubMullet(projections=self.d.projections,
                                   projector=self.projector)

        self.u_shape = num_voxel

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print "%s: %.3f" % (self.id(), t)
        # Clear ASTRA memory
        self.projector.clear()

    def test_initialization(self):
        self.assertTrue(issubclass(type(self.cgm), object))

    def test_g(self):
        g = self.cgm.g
        u_shape = self.cgm.K.volume_shape
        self.assertEqual(g.__class__.__name__, 'ndarray')
        self.assertEqual(np.shape(g), tuple((x - 0 for x in u_shape)))

    def test_f(self):
        f = self.cgm.f
        fl = list(f)
        # ft = tuple(f)
        self.assertEqual(len(fl), len(self.d.shape))
        # self.assertEqual(type(f), list)

    def test_func_du(self):
        func_du = self.cgm.func_du(np.zeros(self.u_shape))
        self.assertEqual(func_du.shape, self.u_shape)
        self.assertTrue(func_du.any() == 0)


class ChambollePockTestCase(unittest.TestCase):
    """Test case for primal-dual method Chambolle-Pock algorithm."""

    def setUp(self):
        # Timing
        self.start_time = time.time()

        # DATA
        self.d = ctdata.sets[14]
        self.d.load()

        # Parameters
        det_row_count, num_proj, det_col_count = self.d.shape
        num_voxel = (det_col_count, det_col_count, det_row_count)
        voxel_size = 2 * self.d.roi_cubic_width_mm / num_voxel[0]
        source_origin = self.d.distance_source_origin_mm / voxel_size
        origin_detector = self.d.distance_origin_detector_mm / voxel_size
        angles = self.d.angles_rad
        det_col_spacing = self.d.detector_width_mm / det_col_count / voxel_size
        det_row_spacing = det_col_spacing

        # PROJECTOR
        self.projector = Projector(
            num_voxel=num_voxel,
            det_row_count=det_row_count, det_col_count=det_col_count,
            source_origin=source_origin, origin_detector=origin_detector,
            det_row_spacing=det_row_spacing, det_col_spacing=det_col_spacing,
            angles=angles)

        # ALGORITHM
        self.pc = ChambollePock(projections=self.d.projections,
                                projector=self.projector)

        self.u_shape = num_voxel

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print "%s: %.3f" % (self.id(), t)
        # Clean ASTRA memory
        self.projector.clear()

    def test_projection_data(self):
        d = self.d.projections
        print 'min:', d.min(), 'max:', d.max(), 'mean:', np.mean(d)
        self.assertTrue(self.d.projections.min() > 0)
        self.assertTrue(self.d.projections.max() < np.inf)

    def test_initialization(self):
        self.assertTrue(self.d.projections.shape > 0)
        self.assertTrue(self.pc.K.volume_data)
        self.pc.K.backward()
        # self.assertEqual(str(self.d.dtype), 'uint16')
        self.assertEqual(str(self.d.dtype), 'float32')
        self.assertEqual(self.pc.K.volume_data.dtype.__str__(), 'float32')
        self.assertTrue(self.pc.K.volume_shape > 0)
        self.assertTrue(issubclass(type(self.pc), object))
        self.assertEqual(self.pc.K, self.projector)

    def test_matrix_norm(self):
        # Start computation of matrix
        num_iter = 2
        mat_norm_list = self.pc.matrix_norm(
            num_iter, vol_init=1, intermediate_results=True)
        self.assertEqual(np.size(mat_norm_list), num_iter)
        # Continue iteration of matrix, starting from the above results
        mat_norm = self.pc.matrix_norm(3, continue_iteration=True)
        self.assertEqual(np.size(mat_norm), 1)
        self.assertTrue(mat_norm > 0)
        self.assertNotEqual(mat_norm_list[-1], mat_norm)

        mat_norm = self.pc.matrix_norm(20, vol_init=1,
                                       intermediate_results=True)
        self.pc.K.clear()
        print mat_norm

    def test_least_squares(self):
        num_iter = 10
        u, p, cpd, l2_atp = self.pc.least_squares(
            num_iterations=num_iter,
            L=363.569641113,
            verbose=True,
            non_negativiy_constraint=False)
        self.assertEqual(u.__class__.__name__, 'ndarray')
        self.assertEqual(cpd.size, num_iter)
        self.assertEqual(l2_atp.size, num_iter)

        self.pc.K.clear()

    def test_least_squares_with_non_negativity_constraint(self):
        num_iter = 4
        u, p, cpd, l2_atp = self.pc.least_squares(
            num_iterations=num_iter,
            L=363.569641113,
            verbose=True,
            non_negativiy_constraint=True)
        self.assertEqual(u.__class__.__name__, 'ndarray')
        self.assertEqual(cpd.size, num_iter)
        self.assertEqual(l2_atp.size, num_iter)

        self.pc.K.clear()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ChambollePockTestCase)
    # unittest.main()
    unittest.TextTestRunner(verbosity=0).run(suite)
