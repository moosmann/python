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
import scipy.io
import matplotlib.pyplot as plt

vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))

# As before, create a sinogram from a phantom

P = scipy.io.loadmat('phantom.mat')['phantom256']
proj_id = astra.create_projector('cuda', proj_geom,vol_geom)
# print('\n CUDA: ', astra.projector.is_cuda(proj_id),
#       astra.projector.projection_geometry(proj_id),
#       )
sinogram_id, sinogram = astra.create_sino(P, proj_id)

plt.gray()
plt.figure(1)
plt.imshow(P)
plt.figure(2)
plt.imshow(sinogram)

# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)
print('rec', rec_id)
r = astra.data2d.get(rec_id)
print('rec', r.min(), r.max())
plt.figure('rec with data none')
plt.imshow(r)
plt.show()
vol_id = astra.data2d.create('-vol', vol_geom, 1)

# Set up the parameters for a reconstruction algorithm using the GPU

cfg = astra.astra_dict('FP_CUDA')
# cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['VolumeDataId'] = vol_id
cfg['ProjectorId'] = proj_id

# Available algorithms:
# SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA (see the FBP sample)
# print('cfg {}'.format(cfg))

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)



# Run 150 iterations of the algorithm
astra.algorithm.run(alg_id, 150)

# Get the result
rec = astra.data2d.get(rec_id)
plt.figure(3)
plt.imshow(rec)
# plt.show()

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)

print('\nFINISHED')