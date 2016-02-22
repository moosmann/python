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


def phantom_ball(num_voxel=None, relative_origin=None, relative_radius=0.3):
    """Create a 3D binary phantom object 'phan' with num_voxel and
    consisting of a ball with radius relative_radius*min(num_voxel)
    and located at num_voxel.*relative_origin.

    Parameters
    ----------
    :rtype : numpy.array
    num_voxel : list of integers of length 1 or 3
        If len(list)==1, num_voxel is extended to length 3 with value num_voxel[0].
    relative_origin : list floats of length 1 or 3 with values in [0,1]
        If len(list)==1, num_voxel is extended to length 3 with value num_voxel[0].
    relative_radius : float

    #>>> phan = phantom_ball([10])
    #>>> print phan.shape
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
    x = np.arange(num_voxel[0])
    y = np.arange(num_voxel[1])
    z = np.arange(num_voxel[2])
    # 3D array
    [x, y, z] = np.meshgrid(y, x, z)
    # centre
    x0 = relative_origin[0] * (num_voxel[0] - 1)
    y0 = relative_origin[1] * (num_voxel[1] - 1)
    z0 = relative_origin[2] * (num_voxel[2] - 1)
    # Radius
    r = relative_radius * np.min(num_voxel)
    # phantom
    a = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 <= r ** 2
    phantom = np.zeros(num_voxel)
    phantom[a] = 1

    return phantom


def conebeam_sim(phantom=None, alg_type='sirt3d', num_iterations=10, padding=(0, 0), scale_factor=1, vol_init=None,
                 gpu_index=0,
                 num_voxel=None, num_pixel=None, num_angles=150):
    """Test conebeam reconstruction of ASTRA.

    conebeam_sim()
    """

    # Default arguments
    if not num_voxel:
        num_voxel = [100, 100, 100]
    if not num_pixel:
        num_pixel = [112, 112]

    # print astra workspace
    if 0:
        astra.algorithm.info()
        astra.data3d.info()
        astra.projector.info()

    if padding[0]:
        pass

    if scale_factor:
        pass

    detector_spacing_x = 1
    detector_spacing_y = 1
    det_row_count = num_pixel[0]
    det_col_count = num_pixel[1]
    angles_rad = np.linspace(0, 2 * np.pi, num_angles, endpoint=True)
    source_origin = 50
    source_det = 50

    print("GPU index: %u" % gpu_index)
    # Create volume geometry
    vol_geom = astra.create_vol_geom(*num_voxel)
    # Create projection geometry
    #    proj_geom = astra.create_proj_geom('parallel3d',
    #                                       detector_spacing_x, detector_spacing_y,
    #                                       det_row_count, det_col_count,
    #                                       angles)
    proj_geom = astra.create_proj_geom('cone',
                                       detector_spacing_x, detector_spacing_y,
                                       det_row_count, det_col_count,
                                       angles_rad,
                                       source_origin, source_det)

    # Create phantom object
    if phantom is None:
        phantom = 1000 / 2 * (phantom_ball(num_voxel, (0.45, 0.5, 0.5), 0.1)
                              + phantom_ball(num_voxel, (0.55, 0.5, 0.5), 0.05))

    t = time.time()
    # Create a sinogram from phantom
    sinogram_id, sinogram = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom, returnData=True, gpuIndex=gpu_index)
    print("Sinogram shape: %u, %u, %u" % (sinogram.shape))
    
    # Set up the parameters for a reconstruction algorithm using the GPU
    rec_id = astra.data3d.create('-vol', vol_geom, vol_init)
    alg_type += '_CUDA'
    alg_type = alg_type.upper()
    cfg = astra.astra_dict(alg_type)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id

    # Use GPU
    cfg['option'] = {}
    cfg['option']['GPUindex'] = gpu_index

    # Iterate algorithm
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iterations)
    rec_vol = astra.data3d.get(rec_id)

    print("Number of projections: %u" % proj_geom['ProjectionAngles'].shape)
    print("Algorithm: %s" % cfg['type'])
    print("Elapsed time: %g s" % (time.time() - t))

    # Clean up.
    # astra.algorithm.delete(alg_id)
    # astra.data2d.delete(rec_id)
    # astra.data2d.delete(sinogram_id)
    # astra.projector.delete(proj_id)

    astra.algorithm.clear()
    astra.data2d.clear()
    astra.projector.clear()

    return rec_vol, sinogram, phantom, angles_rad


from utils import *

# ph = phantom_ball()
nvol = 100
npix = 100
rec, sino, phan, angles = conebeam_sim(
    num_voxel=[nvol, nvol, nvol], num_pixel=[npix, npix],
    num_angles=2 * 180,
    alg_type='sirt3d', num_iterations=100)

s = sino[50, :, :]


# show_slices(sino)
#    show_slices(phan, 'phantom')
#    show_slices(sino, 'sino')
show_slices(phan)
show_slices(rec, 'recon')
# show_slices(rec-phan)

print phan.shape, sino.shape, rec.shape
print phan.min(), phan.max()
print rec.min(), rec.max()

if 0:
    s = sino[50, :, :]
    plt.ion()
    fig = plt.figure('Sinogram')
    ax = fig.add_subplot(211)
    im = ax.imshow(s)
    fig.show()
    im.axes.figure.canvas.draw()


