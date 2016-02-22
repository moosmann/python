#! /usr/bin/env python
"""
Create ASTRA projector object and return explicit matrices of the projector. Test if the adjoint matrix (backprojection)
 is the exact transpose of the forward projector.
"""

__author__ = 'jmoosmann'

import astra
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Create volume geometry
num_voxel = (10, 10, 10)
vol_geom = astra.create_vol_geom(num_voxel)

# Create projection geometry
geometry_type = 'cone'
det_col_count = 10
det_row_count = 10
detector_spacing_x = 1.0 # det_col_spacing
detector_spacing_y = 1.0 # det_row_spacing
angles = np.linspace(0, 2 * np.pi, 18, endpoint=False)
source_origin = 10.0
origin_detector = 10.0
# CONE
# proj_geom = astra.create_proj_geom(geometry_type,
#                                    detector_spacing_x, detector_spacing_y,
#                                    det_row_count, det_col_count,
#                                    angles,
#                                    source_origin, origin_detector)
# FANFLAT

proj_geom = astra.create_proj_geom('fanflat', detector_spacing_x, det_col_count, angles, source_origin, origin_detector)

# print "Proj_geom:", proj_geom
# print "Vol geom:", vol_geom

# Create projector instance
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
# proj_id2 = astra.projector.create(proj_geom)


# astra.matrix.info()

proj_dict = astra.astra_dict(proj_id)
# print "Proj dict:", proj_dict

# Returns ID of sparse matrix of projector
foo = astra.matrix.create
print "foo:", type(foo)

mat_id = astra.projector.matrix(proj_id)
smat = astra.matrix.get(mat_id)
print type(smat), smat.shape, smat.dtype, smat.ndim
smat2 = csr_matrix(smat)
# print smat2
#mat = smat2.todense()
#mat = smat2.toarray()
#print type(mat)

# print plt.get_backend()
# plt.switch_backend('qt4agg')
# plt.figure()
# plt.imshow(smat)
# plt.show()

# Returns whatever
# row = 1
# col = 1
# splat = astra.projector.splat(proj_id, row, col)