#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:17:12 2015

@author: jmoosmann
"""

import numpy as np
import astra
import matplotlib.pyplot as plt

astra.data3d.clear()
astra.algorithm.clear()

# Phantom
nn = 64
phan = np.zeros((nn, nn, nn))
nni = np.round(0.3*nn)
nne = np.round(0.7*nn)
phan[nni:nne, nni:nne, nni:nne] = 1
nni = np.round(0.4*nn)
nne = np.round(0.6*nn)
phan[nni:nne, nni:nne, nni:nne] = 0

# Parameters
num_voxel = (nn, nn, nn)
detector_spacing_x = 1
detector_spacing_y = 1
det_row_count = int(np.round(1.0 * nn))
det_col_count = det_row_count
angles = np.linspace(0, 2 * np.pi, 180)
source_origin = 50
source_det = 50
vol_init=None
gpu_index=0

# Create volume geometry
vol_geom = astra.create_vol_geom(*num_voxel)

# Creat projection geometry
proj_geom = astra.create_proj_geom('cone', detector_spacing_x, detector_spacing_y, det_row_count, det_col_count, angles, source_origin, source_det)
    
# Create a sinogram from phantom
sinogram_id, sinogram = astra.create_sino3d_gpu(phan, proj_geom, vol_geom, returnData=True, gpuIndex=gpu_index)

# Create 3D data object
rec_id = astra.data3d.create('-vol', vol_geom, vol_init)

# SIRT3D
alg_type = 'SIRT3D_CUDA'
cfg_sirt = astra.astra_dict(alg_type)
cfg_sirt['ReconstructionDataId'] = rec_id
cfg_sirt['ProjectionDataId'] = sinogram_id
cfg_sirt['option'] = {}
cfg_sirt['option']['GPUindex'] = gpu_index

# Create algorithm object
alg_id = astra.algorithm.create(cfg_sirt)

# Iterate algorithm
num_iterations = 20
astra.algorithm.run(alg_id, num_iterations)

# Get 3D data
vol_sirt = astra.data3d.get(rec_id)

# FDK
alg_type = 'FDK_CUDA'
cfg_fdk = astra.astra_dict(alg_type)
cfg_fdk['ReconstructionDataId'] = rec_id
cfg_fdk['ProjectionDataId'] = sinogram_id
cfg_fdk['option'] = {}
cfg_fdk['option']['GPUindex'] = gpu_index

# Create algorithm object
alg_id = astra.algorithm.create(cfg_fdk)

# Iterate algorithm
astra.algorithm.run(alg_id, 1)

# Get 3D data
vol_fdk = astra.data3d.get(rec_id)


# Compare
vol = phan
print("PHANTOM: min, max, mean = %6.2g, %6.2g, %6.2g," %(vol.min(), vol.max(), np.mean(vol)))
vol = vol_sirt
print("SIRT3D:  min, max, mean = %6.2g, %6.2g, %6.2g," %(vol.min(), vol.max(), np.mean(vol)))
vol = vol_fdk
print("FDK:     min, max, mean = %6.2g, %6.2g, %6.2g," %(vol.min(), vol.max(), np.mean(vol)))

def show_vol_xyz(vol):
    
    n1 = np.round(vol.shape[0]/2)
    n2 = np.round(vol.shape[1]/2)
    n3 = np.round(vol.shape[2]/2)
    
    cm = plt.cm.Greys
    
    plt.figure()    
    
    plt.subplot(1, 3, 1)
    plt.imshow(vol[n1, :, :],  cmap=cm)
    
    plt.subplot(1, 3, 2)
    plt.imshow(vol[:, n2, :],  cmap=cm)
    
    plt.subplot(1, 3, 3)
    plt.imshow(vol[:, :, n3],  cmap=cm)

#show_vol_xyz(phan)
show_vol_xyz(vol_sirt)
show_vol_xyz(vol_fdk)