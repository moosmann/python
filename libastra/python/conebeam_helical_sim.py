#! /usr/bin/env python
"""
Test module for testing the implementation of the ASTRA toolbox using
its python interface PyAstra.

"""

import astra
import numpy as np
import time
# import matplotlib
# matplotlib.use('QTAgg')
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.pylab as plt
# import os
# import scipy.io as sio
import utils as util


def conebeam_helical_sim(phantom=None, alg_type='sirt3d', num_iterations=10, padding=(0, 0), scale_factor=1,
                         vol_init=None,
                         gpu_index=0,
                         num_voxels=None, num_pixel=None, num_angles=25):
    """Test helical conebeam reconstruction of ASTRA.

    Parameters
    ----------

    num_voxels : int, list of int
    num_voxels[0:1] are the number of transverse voxels, num_voxels[2] is the number of  voxels along the longitudinal
    direction that coincides with the helical (z) axis and along which the sample is translated.

    conebeam_helical_sim()
    """

    # Default arguments
    if not num_voxels:
        num_voxels = [100, 100, 100]
    if not num_pixel:
        num_pixel = [100, 100]

    if padding[0]:
        pass

    if scale_factor:
        pass

    detector_spacing_x = 1
    detector_spacing_y = 1
    det_row_count = num_pixel[0]
    det_col_count = num_pixel[1]
    angles_rad = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    source_origin = 50
    origin_det = 40
    z_max = np.round(num_voxels[2] / 3)
    z_min = -np.round(num_voxels[2] / 3)
    z_range = np.linspace(z_min, z_max, num_angles, endpoint=False)

    # Create phantom object
    if phantom is None:
        phantom = 1000 / 2 * (util.phantom_ball(num_voxels, (0.45, 0.5, 0.5), 0.1)
                              + util.phantom_ball(num_voxels, (0.55, 0.5, 0.5), 0.05))

    # Create volume geometry
    vol_geom = astra.create_vol_geom(*num_voxels)

    # Create helical projection geometry
    cone_vec = np.zeros((num_angles, 12))
    # Source
    cone_vec[:, 0] = np.sin(angles_rad) * source_origin
    cone_vec[:, 1] = -np.cos(angles_rad) * source_origin
    cone_vec[:, 2] = z_range
    # Centre of detector
    cone_vec[:, 3] = -np.sin(angles_rad) * origin_det
    cone_vec[:, 4] = np.cos(angles_rad) * origin_det
    cone_vec[:, 5] = z_range
    # Vector from detector pixel (0,0) to (0,1)
    cone_vec[:, 6] = np.cos(angles_rad) * detector_spacing_x
    cone_vec[:, 7] = np.sin(angles_rad) * detector_spacing_x
    cone_vec[:, 8] = 0
    # Vector from detector pixel (0,0) to (0,1)
    cone_vec[:, 9] = 0
    cone_vec[:, 10] = 0
    cone_vec[:, 11] = detector_spacing_y

    proj_geom = astra.create_proj_geom('cone_vec', det_row_count, det_col_count, cone_vec)

    t = time.time()
    # Create a sinogram from phantom
    sinogram_id, sinogram = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom, returnData=True, gpuIndex=gpu_index)

    # Set up the parameters for a reconstruction algorithm
    rec_id = astra.data3d.create('-vol', vol_geom, vol_init)
    alg_type += '_CUDA'
    alg_type = alg_type.upper()
    cfg = astra.astra_dict(alg_type)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = {}
    cfg['option']['GPUindex'] = gpu_index

    # Iterate algorithm
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iterations)
    rec_vol = astra.data3d.get(rec_id)

    print("GPU index: %u" % gpu_index)
    # print("Number of projections: %u" % proj_geom['ProjectionAngles'].shape)
    print("Algorithm: %s" % cfg['type'])
    print("Elapsed time: %g s" % (time.time() - t))

    astra.algorithm.clear()
    astra.data3d.clear()
    astra.projector.clear()

    return rec_vol, sinogram, phantom, angles_rad, proj_geom


# ph = util.phantom_ball()
nvol = 100
npix = 100
rec, sino, phan, angles, p = conebeam_helical_sim(
    num_voxels=[nvol, nvol, nvol], num_pixel=[npix, npix],
    num_angles=2 * 180,
    alg_type='sirt3d', num_iterations=100)

s = sino[50, :, :]

import matplotlib.pyplot as plt
util.show_slices(sino, 'projection data')
util.show_slices(rec, 'reconstruction')
plt.show()


print 'Shape of phantom, sino, reco:', phan.shape, sino.shape, rec.shape
print 'Min, Max phantom:', phan.min(), phan.max()
print 'Min, Max reco:', rec.min(), rec.max()

if 0:
    import matplotlib.pyplot as plt

    s = sino[50, :, :]
    plt.ion()
    fig = plt.figure('Sinogram')
    ax = fig.add_subplot(211)
    im = ax.imshow(s)
    fig.show()
    im.axes.figure.canvas.draw()


# if __name__ == "__main__":
#    import doctest
#
#    doctest.testm
