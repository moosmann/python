#! /usr/bin/env python
"""
Test module for testing the implementation of the ASTRA toolbox using
its python interface PyAstra.

"""

import astra
# import numpy as np
import time
# import matplotlib
# matplotlib.use('QTAgg')
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import matplotlib.pylab as plt
# import os
# import scipy.io as sio
import ctdata


def conebeam_test(data_obj=ctdata.Data(), alg_type='fdk', num_iterations=10, padding=None, scale_factor=1,
                  vol_init=None,
                  gpu_index=0,
                  num_voxel=None):
    """Test conebeam reconstruction of ASTRA.

    conebeam_sim()
    """

    t = time.time()

    # Default arguments
    if not num_voxel:
        num_voxel = [100, 100, 100]

    if not padding:
        padding = 0

    if scale_factor:
        pass

    # Extract paramaters from data class
    detector_spacing_x = 1
    detector_spacing_y = 1
    det_row_count = data_obj.shape[0]
    det_col_count = data_obj.shape[2] + 2 * padding
    angles_rad = data_obj.angles__rad
    source_origin = data_obj.distance_source_origin__mm
    source_det = data_obj.distance_source_detector__mm

    # Create volume geometry
    vol_geom = astra.create_vol_geom(*num_voxel)

    # Create projection geometry
    proj_geom = astra.create_proj_geom('cone',
                                       detector_spacing_x, detector_spacing_y,
                                       det_row_count, det_col_count,
                                       angles_rad,
                                       source_origin, source_det)

    # Create a sinogram from phantom
    sinogram = np.pad(data_obj.projections, ((0, 0), (0, 0), (padding, padding)), mode='edge')
    sino_id = astra.data3d.create('-sino', proj_geom, sinogram)

    # Set up the parameters for a reconstruction algorithm using the GPU
    rec_id = astra.data3d.create('-vol', vol_geom, vol_init)
    alg_type += '_CUDA'
    alg_type = alg_type.upper()
    cfg = astra.astra_dict(alg_type)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id

    # Use GPU
    cfg['option'] = {}
    cfg['option']['GPUindex'] = gpu_index

    # Iterate algorithm
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iterations)
    rec_vol = astra.data3d.get(rec_id)

    print("Algorithm: %s" % cfg['type'])
    print("Elapsed time: %g s" % (time.time() - t))

    # Clean up.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sino_id)

    return rec_vol, sinogram


from utils import *
# from ctdata import *

if 1:
    # del(data)
    # if 'data' not in locals():
    #     print("load data set")
    #     data = ctdata.sets[13]
    data = ctdata.sets[13]
    data.set_permute_order((1, 2, 0))
    d = data.load()
    data.filter_inf()
    data.filter_nan()
    rec, sino = conebeam_test(data, alg_type='fdk', padding=300)
    show_slices(rec)
    print rec.min(), rec.max(), data.angles__rad[-1]
    plt.show()


# if __name__ == "__main__":
#    import doctest
#
#    doctest.testm
