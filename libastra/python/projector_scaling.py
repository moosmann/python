import astra
import numpy as np
import matplotlib
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import ainfo

matplotlib.use("qt4agg")
import matplotlib.pyplot as plt


# PHANTOM
dim = 100
vol_width_mm = 50.0  # mm
voxel_size_mm = vol_width_mm / dim
vshape = 2 * (dim,)
phan1 = np.ones(vshape)
phanb = np.zeros(vshape)
x0 = int(np.floor(dim / 4.0))
x1 = int(np.ceil(3 * dim / 4.0))
phanb[x0:x1, :] = 1
gpu_index = 0
print('2D PHANTOM: parallel beam')
print('  Unit: shape = {0}, mean = {1}, min = {2}, max = {3}'.format(
    vshape, phan1.mean(), phan1.min(), phan1.min()))
print('  Binary: shape = {0}, mean = {1}, min = {2}, max = {3}'.format(
    vshape, phanb.mean(), phanb.min(), phanb.min()))
print('  Voxel size = {0} mm, volume width = {1} mm'.format(
    voxel_size_mm, vol_width_mm))

vol_geom = astra.create_vol_geom(vshape)
proj_id = astra.create_projector('cuda', astra.create_proj_geom(
    'parallel', 1.0, vshape[0], np.linspace(0, np.pi, 200, False)), vol_geom)

def pid(number_of_angles):
    """Return projector ID.
    :param number_of_angles:
    :return:
    """
    return astra.create_projector('cuda', astra.create_proj_geom('parallel',
    1.0, vshape[0], np.linspace(0, np.pi, number_of_angles, False)), vol_geom)


def fp0(image, det_col=111, num_angles=222, voxel_size_mm=1):
    """Wrapper for astra forward projector

    :param image:
    :return: sinogram
    """

    vol_geom = astra.create_vol_geom(image.shape)
    proj_id = astra.create_projector(
        'cuda',
        astra.create_proj_geom('parallel', 1.0, det_col,
                               np.linspace(0, np.pi, num_angles, False)),
        vol_geom)

    sino_id, sino = astra.create_sino(image, proj_id)
    sino *= voxel_size_mm

    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)

    return sino

def fp(image, projector_id):
    """Wrapper for astra forward projector

    :param image:
    :param projector_id:
    :return: sinogram
    """
    sino_id, sino = astra.create_sino(image, projector_id)
    sino *= voxel_size_mm

    sino /= sino.shape[0]

    astra.data2d.delete(sino_id)

    return sino


def bp0(sino, det_col=111, num_angles=222, voxel_size_mm=1):
    """Wrapper for astra forward projector

    :param sino:
    :param projector_id:
    :return: backprojected sinogram
    """

    vol_geom = astra.create_vol_geom(sino.shape)
    proj_id = astra.create_projector(
        'cuda',
        astra.create_proj_geom('parallel', 1.0, det_col,
                               np.linspace(0, np.pi, num_angles, False)),
        vol_geom)


    rec_id, backprojection = astra.create_backprojection(sino * voxel_size_mm,
                                                   proj_id)

    # rec_id, backprojection = astra.create_backprojection(sino, projector_id)

    # backprojection /= sino.shape[0]
    # backprojection *= np.pi
    astra.data2d.delete(rec_id)
    astra.projector.delete(proj_id)

    return backprojection

def bp(sino, projector_id):
    """Wrapper for astra forward projector

    :param sino:
    :param projector_id:
    :return: backprojected sinogram
    """

    rec_id, backprojection = astra.create_backprojection(sino * voxel_size_mm,
                                                   projector_id)

    backprojection /= sino.shape[0]
    # backprojection *= np.pi
    astra.data2d.delete(rec_id)

    return backprojection


sinob = fp(phanb, proj_id)
sino1 = fp(phan1, proj_id)

# _, rec0 = astra.create_reconstruction('BP_CUDA', proj_id,
# sino0*voxel_size_mm, 1)
rec1 = bp(sino1, proj_id)
recb = bp(sinob, proj_id)

print('SINOGRAM')
ndim = 100
s0 = fp0(np.ones((ndim, ndim)),det_col=88, num_angles=77, voxel_size_mm=1)
ndim = 50
s1 = fp0(np.ones((ndim, ndim)),det_col=88, num_angles=99, voxel_size_mm=2)
print('  Scaling with number of voxels keeping volume size fixes, '
      'also change number of angles')
print('  100 pixesl: {0}, 50 pixels: {1}'.format(s0[0, :].max(),
                                                 s1[0, :].max()))
print('  unit: mean, min, max =  %f, %f, %f' % (
    sino1.mean(), sino1.min(), sino1.max()))
print('  binary: mean, min, max = =  {0}, {1}, {2}'.format(
    sinob.mean(), sinob.min(), sinob.max()))

print('ADJOINT')
x = phan1
y = np.ones(sino1.shape)
Ady = bp(y, proj_id)
# Ady2 = bp0(y,)
Ax = fp(np.ones(phan1.shape), proj_id)
n1 = np.sum(np.ravel(Ax * y))
n2 = np.sum(np.ravel(Ady * x))
s = n1 / n2
print('  x[:] = 1, y[:] = 1, x.size = {0}, y.size = {1}'.format(x.size,
                                                                y.size))
print '  <A x, y> = <x, A^* y> : {0} = {1}'.format(n1, n2)
print '  s = <A x, y>  / <x, A^* y> = {0} with A^* = s B. s - 1 = {' \
      '1}'.format(s, s - 1)
print '  Error: abs = {0}, rel = {1}'.format(n1 - n2,
                                             (n1 - n2) / (n1 + n2) * 2)
s1 = np.linalg.norm(sino1.ravel()) ** 2 / np.sum(np.ravel(rec1 * phan1))
print '  Let y = A x with adjoint A^* = s1 B and ASTRA back-projector B:'
print('  s1 = {0}, s-s1 = {1}, s/s1 - 1 = {2}'.format(s1, s-s1, s/s1-1))

p0 = sinob[0, :]
p90 = sinob[np.round(sinob.shape[0] / 2), :]
print('PROJECTION VALUES at 0 and 90 degrees')
print('  binary: {0} at 0 deg, {1} at 90 deg'.format(p0.max(), p90.max()))

print('BACKPROJECTION OF UNIT SINOGRAM (/ voxel size)')
rec100 = bp(np.ones((100, dim))/voxel_size_mm, pid(100))
rec200 = bp(np.ones((200, dim))/voxel_size_mm, pid(200))
print('  100 angles: {0}, {1}'.format(rec100.min(), rec100.max()))
print('  200 angles: {0}, {1}'.format(rec200.min(), rec200.max()))

# Geometry
detector_spacing = 1.0
det_row_count = 100
source_origin = 100.0
origin_detector = 10.0

# Create projection geometry
# proj_geom = astra.create_proj_geom('fanflat', detector_spacing,
# det_row_count, angles, source_origin, origin_detector)
# proj_geom = astra.create_proj_geom('parallel', 1.0, vshape[0], np.linspace(
#     0, np.pi, 8, False))
# proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,
# np.pi,180,False))
#
# proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
#
# sinogram_id, sinogram = astra.create_sino(phan, proj_id)

# Allocate ASTRA memory for volume data
# volume_id = astra.data2d.create('-vol', vol_geom)

# print proj_geom
# Allocate ASTRA memory for projection data
# proj_id = astra.data2d.create('-sino', proj_geom)
# proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
# _, proj = astra.create_sino(phan, proj_id)

# astra.data2d.store(volume_id, phan)
#
# # Create algorithm object
# cfg = astra.astra_dict('FP_CUDA')
# cfg['option'] = {'GPUindex': gpu_index}
# cfg['ProjectionDataId'] = proj_id
# cfg['VolumeDataId'] = volume_id
# fp_id = astra.algorithm.create(cfg)
#
# # Run algorithm
# astra.algorithm.run(fp_id)
#
# # Retrieve projection data from ASTRA memory
# proj = astra.data2d.get(proj_id)
#
# print proj.shape
#
# # Store projection data in ASTRA memory
# astra.data2d.store(proj_id, proj)
#
# # Create algorithm object
# cfg = astra.astra_dict('BP_CUDA')
# cfg['option'] = {'GPUindex': gpu_index}
# cfg['ProjectionDataId'] = proj_id
# cfg['ReconstructionDataId'] = volume_id
# bp_id = astra.algorithm.create(cfg)
#
# # Run algorithms
# astra.algorithm.run(bp_id)
#
# # Retrieve projection from ASTRA memory
# rec = astra.data2d.get(volume_id)

# Plots and images
cm = plt.get_cmap('Greys')
fig = plt.figure('Binary phantom')


def add_colorbar(image):
    """Add colorbar to subplot image.

    :type image: matplotlib.image.AxesImage
    :param image: axes.imshow  instance
    :return: cbar, cax, divider
    """

    # Get axes from image instance: matplotlib.axes._subplots.AxesSubplot
    ax = image.get_axes()

    # Create divider for existing axes instance
    divider = make_axes_locatable(ax)

    # Append axes to the right of ax3, with 20% width of ax3
    cax = divider.append_axes("right", size="10%", pad=0.05)

    # Change colorbar ticks format
    scalform = ticker.ScalarFormatter()
    scalform.set_scientific(True)
    scalform.set_powerlimits((-1, 1))

    # Create colorbar in the appended axes
    # cbar = plt.colorbar(image, cax=cax, format="%.2f")
    # cbar = plt.colorbar(image, cax=cax, ticks=ticker.MultipleLocator(0.2),
    #                     format="%.2f")
    cbar = plt.colorbar(image, cax=cax, format=scalform)

    return cbar, cax, divider

ax1 = fig.add_subplot(2, 3, 1)
im1 = ax1.imshow(phanb, cmap=cm, interpolation='none')
ax1.set_title('image[vert, hor]')
ax1.set_xlabel('dim 1: hor')
ax1.set_ylabel('dim 0: vert')
add_colorbar(im1)

ax2 = fig.add_subplot(2, 3, 2)
im2 = ax2.imshow(sinob, cmap=cm, interpolation='none')
ax2.set_title('sinogram[angle, pixel]')
ax2.set_xlabel('dim 1: pixel')
ax2.set_ylabel('dim 0: angle')
add_colorbar(im2)

ax3 = fig.add_subplot(2, 3, 3)
x = np.arange(0, p0.size)
ax3.plot(x, p0, x, p90)
ax3.set_title('sinogram[angle, :]')
ax3.legend(['0 deg', '90 deg'])

ax4 = fig.add_subplot(2, 3, 4)
im4 = ax4.imshow(recb, cmap=cm, interpolation='none')
ax4.set_title('bin. sino bp[vert, hor]')
ax4.set_xlabel('dim 1: hor')
ax4.set_ylabel('dim 0: vert')
add_colorbar(im4)

ax5 = fig.add_subplot(2, 3, 5)
im5 = ax5.imshow(rec100, cmap=cm, interpolation='none')
ax5.set_title('unit sino bp[vert, hor], 100 angl')
ax5.set_xlabel('dim 1: hor')
ax5.set_ylabel('dim 0: vert')
add_colorbar(im5)


plt.tight_layout()
plt.show()

# Free memory
astra.data2d.clear()
astra.data3d.clear()
astra.projector.clear()
astra.algorithm.clear()
