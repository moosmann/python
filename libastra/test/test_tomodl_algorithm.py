"""
Test cases for ODL algorithms instances.
"""

import unittest
import time
import numpy as np
import ctdata
from pyastra import TomODLGeometry
from pyastra import TomODLProjector
from tomodl_algorithm import TomODLChambollePock
import odl
from odl import Rn
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt


class ChambollePockParallelTestCase(unittest.TestCase):
    """Test case for primal-dual method Chambolle-Pock algorithm for ODL"""

    def setUp(self):
        # Timing
        self.start_time = time.time()

        # Geometry class
        self.geom_class = TomODLGeometry

        # Projector class
        self.projector_class = TomODLProjector

        # Algorithm class
        self.cp_class = TomODLChambollePock

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print "%s: %.3f" % (self.id(), t)
        plt.show()

    def test_projector(self):

        data = ctdata.sets['parallel']
        data.load()
        print ' Data detector width:', data.detector_width_mm
        print ' Data projections shape:', data.projections.shape

        geom = self.geom_class(data, 2*(100,), 2*(100,))
        print ' Detector pixel width:', geom.det_col_spacing, geom.det_row_spacing

        projector = self.projector_class(geom)

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

        # import scipy.io as sio
        # sio.savemat(self.matfile, {'parallel_fbp': rec})
        # plt.imshow(rec, cmap=plt.cm.Greys)
        # plt.show()

        projector.clear_astra_memory()

    def test_algorithm_init(self):

        data = ctdata.sets['parallel']
        data.normalize = True
        data.load()
        print ' Data detector width:', data.detector_width_mm
        print ' Data projections shape:', data.projections.shape
        print ' Raw data: min, max = ', data.raw_data_min, data.raw_data_max
        print ' Data projections: min, max = ', data.projections.min(), \
            data.projections.max()

        geom = self.geom_class(data, 2*(150,), 2*(110,))
        print ' Detector pixel width:', geom.det_col_spacing, geom.det_row_spacing

        # projector = self.projector_class(geom)

        # phantom data
        rn_proj = Rn(geom.proj_size).element(data.projections.ravel())
        cp = self.cp_class(geom, rn_proj)

        mat_norm0 = cp.matrix_norm(iterations=4, vol_init=1,
                                   intermediate_results=True)
        print mat_norm0
        mat_norm0 = cp.matrix_norm(iterations=20, vol_init=1,
                                   intermediate_results=False)

        mat_norm_tv = cp.matrix_norm(iterations=20, vol_init=1, tv_norm=True,
                                     intermediate_results=False)
        print('Matrix norm: LS: {0}, TV: {1}'.format(mat_norm0, mat_norm_tv))

        # plt.imshow(data.projections)

        # Least squares
        num_iter = 200
        print cp.adj_scal_fac
        cp.adjoint_scaling_factor()
        print cp.adj_scal_fac

        print cp.adj_scal_fac
        cp.least_squares(num_iter, L=mat_norm0, non_negativiy_constraint=True,
                         tv_norm=False, verbose=True)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        ChambollePockParallelTestCase)
    unittest.TextTestRunner(verbosity=0).run(suite)
    # unittest.main()
