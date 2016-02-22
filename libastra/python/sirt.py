# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:46:52 2015

@author: jmoosmann
"""

import numpy as np
# from jmutility import Cuboid
from pyastra import Projector
# import jmutility as jm
import ctdata
import matplotlib
matplotlib.use("qt4agg")
from matplotlib import pyplot as plt
# from numba import jit

__metaclass__ = type

# SIRT
d = ctdata.sets[14]
d.load()
# show_slices(d.projections)
# Parameters
num_iter = 50
det_row_count, num_proj, det_col_count = d.shape
num_voxel = (det_col_count, det_col_count, det_row_count)
# voxel_size = 1
voxel_size = 2 * d.roi_cubic_width_mm / num_voxel[0]
source_origin = d.distance_source_origin_mm / voxel_size
origin_detector = d.distance_origin_detector_mm / voxel_size
angles = d.angles_rad
det_col_spacing = d.detector_width_mm / det_col_count / voxel_size
det_row_spacing = det_col_spacing

# Projector instance
p = Projector(num_voxel=num_voxel,
              det_row_count=det_row_count, det_col_count=det_col_count,
              source_origin=source_origin, origin_detector=origin_detector,
              det_row_spacing=det_row_spacing, det_col_spacing=det_col_spacing,
              angles=angles)

# Create row sums of system matrix
p.set_volume_data(1)
p.forward()
# rs = p.projection_data.copy()
row_sum = p.projection_data
row_sum[row_sum > 0] = 1.0 / row_sum[row_sum > 0]

# Create colum sums of system matrix
p.set_projection_data(1)
p.backward()
# cs = p.volume_data.copy()
col_sum = p.volume_data
col_sum[col_sum > 0] = 1.0 / col_sum[col_sum > 0]

# Store projection data in ASTRA memory
p.set_projection_data(d.projections)

# Initialize volume with zeros
rec = np.zeros(num_voxel)

# Initialize norm vector
rec_norm = np.zeros(num_iter)
res_norm = np.zeros(num_iter)

# SIRT
fig = plt.figure('SIRT iteration')
cm = plt.cm.Greys
xx = np.arange(num_iter)
numvoxs = np.prod(num_voxel)
n1, n2, n3 = np.rint(np.array(num_voxel) / 2.0)

# Create subplots
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
annotation = plt.annotate(0, xy=(0.5, 0.5), xycoords='figure fraction')

# Show orthogonal slices
ax1.imshow(rec[n1, :, :], cmap=cm)
ax1.set_title("yz-slice %u" % n1)
ax1.axis('off')
ax2.imshow(rec[:, n2, :], cmap=cm)
ax2.set_title("xz-slice %u" % n2)
ax2.axis('off')
ax3.imshow(rec[:, :, n3], cmap=cm)
ax3.set_title("xy-slice %u" % n3)
ax3.axis('off')

# Plot l2-norms
l1, = ax4.plot(xx, res_norm, 'r')
ax4.set_title("l2-norm p. pixels vs iteration")
ax5 = ax4.twinx()
l2, = ax5.plot(xx, rec_norm, 'b')
plt.legend([l1, l2], ['residual', 'reconstruction'], loc='center right')
#ax4.legend(['residual'], loc='upper left')
#ax5.legend(['reconstruction'], loc='lower left')


# plt.ion()
# plt.show()

print " Iteration: "
for nn in range(num_iter):
    print nn,

    annotation.remove()
    annotation = plt.annotate(nn, xy=(0.5, 0.5), xycoords='figure fraction')

    # 1: Forward projection of current reconstruction
    p.set_volume_data(rec)
    p.forward()

    # residual norm
    res_norm[nn:] = np.linalg.norm(np.ravel(d.projections - p.projection_data), ord=2) / numvoxs
    l1.set_data(xx, res_norm)

    if nn > 0:
        ax4.set_ylim([res_norm.min(), res_norm.max()])

    # 2: Subtract result from data
    # 3: Multiply with row sum of system matrix
    # 4: Back projection of result
    p.set_projection_data(row_sum * (d.projections - p.projection_data))
    p.backward()

    # 5: Multiply with column sum of system matrix and update volume
    rec += col_sum * p.volume_data

    # Volume norm
    rec_norm[nn:] = np.linalg.norm(np.ravel(rec), ord=2) / numvoxs

    l2.set_data(xx, rec_norm)
    if nn > 0:
        ax5.set_ylim([rec_norm.min(), rec_norm.max()])

    # Display slices
    # plt.ion()
    plt.hold(True)
    ax1.imshow(rec[n1, :, :], cmap=cm)
    ax2.imshow(rec[:, n2, :], cmap=cm)
    ax3.imshow(rec[:, :, n3], cmap=cm)

    # plt.draw()
    plt.pause(0.1)

# plt.draw()
plt.show(block=False)
# plt.close()
plt.close('all')

p.clear()
