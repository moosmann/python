#-----------------------------------------------------------------------
#Copyright 2013 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/pyastratoolbox/
#
#
#This file is part of the Python interface to the
#All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").
#
#The Python interface to the ASTRA Toolbox is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#The Python interface to the ASTRA Toolbox is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with the Python interface to the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------

import astra
import numpy as np
# import matplotlib
#matplotlib.use("qt4agg")
import matplotlib.pyplot as plt

# Create a basic 256x256 square volume geometry
ndim = (256, 128)
vol_geom = astra.create_vol_geom(*ndim)
vol_geom = astra.create_vol_geom(256, 128,

                                 -128, 128,
                                 -64, 64,
                                 )
print(vol_geom)

# function astra_create_proj_geom .
proj_geom = astra.create_proj_geom('parallel', 1.0, 384,
                                   np.linspace(0, np.pi, 180, False))

# Load a 256x256 phantom image
import scipy.io
P = scipy.io.loadmat('phantom.mat')['phantom256']
P = P[0:ndim[0], 0:ndim[1]]
P[0:128, :] = 0
P[:, 0:64] = 0
print('P:', P.shape)

# Create a sinogram using the GPU.
# Note that the first time the GPU is accessed, there may be a delay
# of up to 10 seconds for initialization.
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sinogram_id, sinogram = astra.create_sino(P, proj_id)
print('sinogram:', sinogram.shape)
#
# proj_id = astra.create_projector('line', proj_geom,vol_geom)
# sinogram_id, sinogram = astra.create_sino(P, proj_id)
# print('sinogram:', sinogram.shape)


# bp_id, bp = astra.create_backprojection(np.ones(sinogram.shape), proj_id)
# bp_id, bp = astra.create_backprojection(sinogram, proj_id)
# print('bp', bp.shape)
#
# # ODL test
# ndim = (100, 110)
# vol_geom = astra.create_vol_geom(*ndim)
# proj_geom = astra.create_proj_geom('parallel', 1.0, 200,
#                                    np.linspace(0, 2*np.pi, 180, False))
# proj_id = astra.create_projector('linear',proj_geom,vol_geom)
# P = np.zeros(ndim, dtype=np.float32)
# P[20:30, 20:30] = 1
# gpuIndex = None
# sinogram_id, sinogram = astra.create_sino(P, proj_id)
#
#
# # volume_id = astra.data2d.create('-vol', vol_geom, P)
# volume_id = astra.data2d.link('-vol', vol_geom, P.astype(np.float32))
# # sino_id = astra.data2d.create('-sino', proj_geom, 0)
# sino_id = astra.data2d.link('-sino', proj_geom, np.zeros(
#     sinogram.shape).astype(np.float32))
#
# if astra.projector.is_cuda(proj_id):
#     algString = 'FP_CUDA'
# else:
#     algString = 'FP'
#
# cfg = astra.astra_dict(algString)
# cfg['ProjectorId'] = proj_id
# if gpuIndex is not None:
#     cfg['option'] = {'GPUindex': gpuIndex}
# cfg['ProjectionDataId'] = sino_id
# cfg['VolumeDataId'] = volume_id
# alg_id = astra.algorithm.create(cfg)
# astra.algorithm.run(alg_id)
# astra.algorithm.delete(alg_id)
#
# astra.data2d.delete(volume_id)
#
# # sino = astra.data2d.get(sino_id)
# sino = astra.data2d.get_shared(sino_id)
# # print(sino.flags, sino.dtype)
#
#
# bp_id, bp = astra.create_backprojection(sino, proj_id)
#
#
# plt.figure('sino')
# plt.imshow(sino)
# # plt.show()
#
#
# plt.figure('backprojection from flat sinogram')
# plt.imshow(bp)
# # plt.show()
#
#
#
# dl = astra.data2d.link('-vol', vol_geom, P.astype(np.float32))
# # print(type(dl))
# # print(dl)
#
#
# # plt.gray()
# # plt.figure('phantom')
# # plt.imshow(P)
# # plt.figure('sinogram')
# # plt.imshow(sinogram)
# # plt.figure('backprojection from flat sinogram')
# # plt.imshow(bp)
#
# # plt.show()
#
#
# # Free memory
# astra.data2d.delete(sinogram_id)
# astra.projector.delete(proj_id)
#
# print('\nFINISHED')
