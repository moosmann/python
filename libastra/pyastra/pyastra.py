"""
ASTRA wrapper for ODL.
"""

from __future__ import division
from odl import Rn, Operator
import astra
import numpy as np
from utils import Cuboid  # for doctests only
import ctdata

# from numba import jit
# from RL.utility.testutils import Timer


class TomODLGeometry(object):

    def __init__(self, loaded_data_object=ctdata.Data(),
                 vol_shape=(100, 100, 100),
                 vol_width_mm=(100, 100, 100)
                 ):
        self.data = loaded_data_object
        self.vol_shape = vol_shape
        self.vol_width_mm = vol_width_mm

    @property
    def geom_type(self):
        return self.data.geometry_type

    @property
    def det_col_count(self):
        return self.data.projections.shape[-1]

    @property
    def det_row_count(self):
        if self.proj_ndim == 2:
            return 1
        elif self.proj_ndim == 3:
            return self.data.projections.shape[0]
        else:
            raise Exception("Number of dimensions 'proj_dim' not supported.")

    @property
    def det_col_spacing(self):
        det_width = self.data.detector_width_mm
        if np.size(det_width) == 1:
            det_col_spacing = det_width / self.det_col_count
        elif np.size(det_width) == 2:
            det_col_spacing = det_width[0] / self.det_col_count
        # Normalize to voxel width
        return det_col_spacing / self.voxel_width[0]

    @property
    def detector_spacing_x(self):
        return self.det_col_spacing

    @property
    def det_row_spacing(self):
        det_width = self.data.detector_width_mm
        if np.size(det_width) == 1:
            det_row_spacing = det_width / self.det_row_count
        elif np.size(det_width) == 2:
            det_row_spacing = det_width[1] / self.det_row_count
        else:
            raise Exception("Unsupported detector dimension.")
        # Normalize to voxel width
        det_row_spacing /= self.voxel_width[0]
        return det_row_spacing

    @property
    def detector_spacing_y(self):
        return self.det_row_spacing

    @property
    def vol_size(self):
        return np.prod(self.vol_shape)

    @property
    def vol_ndim(self):
        return len(self.vol_shape)

    @property
    def voxel_width(self):
        return [wi/ni for (wi, ni) in zip(self.vol_width_mm, self.vol_shape)]

    @property
    def proj_size(self):
        return np.prod(self.proj_shape)

    @property
    def proj_shape(self):
        return self.data.projections.shape

    @property
    def proj_ndim(self):
        return len(self.data.projections.shape)

    @property
    def angles(self):
        if self.proj_ndim == 2:
            num_proj = self.proj_shape[0]
        elif self.proj_ndim == 3:
            num_proj = self.proj_shape[1]
        else:
            raise Exception("Usupported number of dimensions 'proj_ndim'")
        return np.linspace(0, self.data.angle_range_rad, num_proj,
                           endpoint=False)

    @property
    def source_origin(self):
        return self.data.distance_source_origin_mm / self.voxel_width[0]

    @property
    def origin_detector(self):
        return self.data.distance_origin_detector_mm / self.voxel_width[0]


class TomODLProjector(object):
    """Tomographic forward and backward projector for ODL using ASTRA."""

    type2d = ['parallel', 'fanflat', 'fanflat_vec']
    type3d = ['parallel3d', 'cone', 'cone_vec', 'parallel3d_vec']

    def __init__(self, geometry=TomODLGeometry()):
        self.geom = geometry
        # Geometries
        self.vol_geom = self._vol_geom()
        self.proj_geom = self._proj_geom()
        # Data IDs
        self.vol_id = self._vol_id()
        self.proj_id = self._proj_id()
        # Algorithms IDs
        self.gpu_index = 0
        self.fp_id = self._fp_id()
        self.bp_id = self._bp_id()
        self.fbp_id = self._fbp_id()
        # Scaling
        self.scaling = 1

    def _vol_geom(self):
        """Create ASTRA volume geometry object according to geometry."""

        vol_shape = self.geom.vol_shape

        assert len(vol_shape) in (2, 3)

        return astra.create_vol_geom(vol_shape)

    def _proj_geom(self, vec=None):
        """Create ASTRA volume geometry object according to geometry."""

        geom = self.geom

        geom_type = geom.geom_type

        assert isinstance(geom_type, str)
        geom_type = geom_type.lower()

        # A parallel projection geometry.
        if geom_type == 'parallel':
            proj_geom = astra.create_proj_geom(
                'parallel',
                geom.detector_spacing_x,
                geom.det_col_count,
                geom.angles)

        # A fan-beam projection geometry.
        elif geom_type == 'fanflat':
            proj_geom = astra.create_proj_geom(
                'fanflat',
                geom.detector_spacing_x,
                geom.det_col_count,
                geom.angles,
                geom.source_origin,
                geom.origin_detector)

        # A fan-beam projection geometry.
        elif geom_type == 'fanflat_vec':
            proj_geom = astra.create_proj_geom(
                'fanflat_vec',
                geom.det_col_count,
                vec)

        # A parallel projection geometry.
        elif geom_type == 'parallel3d':
            proj_geom = astra.create_proj_geom(
                'parallel3d',
                geom.detector_spacing_x,
                geom.detector_spacing_y,
                geom.det_row_count,
                geom.det_col_count,
                geom.angles)

        # A parallel projection geometry.
        elif geom_type == 'parallel3d_vec':
            proj_geom = astra.create_proj_geom(
                'parallel3d',
                geom.det_row_count,
                geom.det_col_count,
                vec)

        # A cone-beam projection geometry.
        elif geom_type == 'cone':
            proj_geom = astra.create_proj_geom(
                'cone',
                geom.detector_spacing_x,
                geom.detector_spacing_y,
                geom.det_row_count,
                geom.det_col_count,
                geom.angles,
                geom.source_origin,
                geom.origin_detector)

        # A cone-beam projection geometry.
        elif geom_type == 'cone_vec':
            proj_geom = astra.create_proj_geom(
                'cone',
                geom.det_row_count,
                geom.det_col_count,
                vec)

        else:
            raise Exception('Unkown type of geometry.')

        return proj_geom

    def _vol_id(self):
        """Returns ASTRA volume id according to geometry."""

        geom_type = self.geom.geom_type.lower()

        type2d = ['parallel', 'fanflat', 'fanflat_vec']
        type3d = ['parallel3d', 'cone', 'cone_vec', 'parallel3d_vec']

        if geom_type in type2d:
            vol_id = astra.data2d.create('-vol', self.vol_geom)
        elif geom_type in type3d:
            vol_id = astra.data3d.create('-vol', self.vol_geom)
        else:
            raise Exception('Unknown type geometry.')

        return vol_id

    def _proj_id(self):
        """Returns ASTRA projection id according to geometry."""

        geom_type = self.geom.geom_type.lower()

        if geom_type in self.type2d:
            proj_id = astra.data2d.create('-sino', self.proj_geom)
        elif geom_type in self.type3d:
            proj_id = astra.data3d.create('-sino', self.proj_geom)
        else:
            raise Exception('Unknown type geometry.')

        return proj_id

    def _fp_id(self):
        """Create algorithms object of forward projection."""

        geom_type = self.geom.geom_type.lower()
        cfg = None

        if geom_type in self.type2d:
            cfg = astra.astra_dict('FP_CUDA')
        elif geom_type in self.type3d:
            cfg = astra.astra_dict('FP3D_CUDA')
        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = self.proj_id
        cfg['VolumeDataId'] = self.vol_id

        return astra.algorithm.create(cfg)

    def _bp_id(self):
        """Create algorithms object of back-projection."""

        geom_type = self.geom.geom_type.lower()
        cfg = None

        if geom_type in self.type2d:
            cfg = astra.astra_dict('BP_CUDA')

        elif geom_type in self.type3d:
            cfg = astra.astra_dict('BP3D_CUDA')
        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = self.proj_id
        cfg['ReconstructionDataId'] = self.vol_id

        return astra.algorithm.create(cfg)

    def _fbp_id(self):
        """Create algorithms object of back-projection."""

        geom_type = self.geom.geom_type.lower()

        if geom_type == 'parallel3d':
            return None

        cfg = None

        if geom_type in self.type2d:
            cfg = astra.astra_dict('FBP_CUDA')
        elif geom_type == 'cone':
            cfg = astra.astra_dict('FDK_CUDA')

        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = self.proj_id
        cfg['ReconstructionDataId'] = self.vol_id

        return astra.algorithm.create(cfg)

    def _store_volume(self, rn_vector=Rn(1).element(1)):
        """Store volume data of Rn vector in ASTRA memory.

        Parameters
        ----------
        :type rn_vector: odl.space.cartesian.Rn
        :param rn_vector: Vector in Rn containing 2D or 3D volume data.
        """

        geom = self.geom

        if geom.geom_type in self.type2d:
            astra.data2d.store(
                self.vol_id, rn_vector.data.reshape(geom.vol_shape))

        elif geom.geom_type in self.type3d:
            astra.data3d.store(
                self.vol_id, rn_vector.data.reshape(geom.vol_shape))
        else:
            raise Exception('Unknown geometry type.')

    def _get_projections(self):
        """Returns Rn vector containing the 2D or 3D projection data
        sinograms).

        Add description of order of dimensions.

        Returns
        -------
        :rtype: odl.space.cartesian.Rn
        :returns:  Vector in Rn containing 2D or 3D projection data.
        """

        geom = self.geom

        if geom.geom_type in self.type2d:
            return Rn(geom.proj_size).element(self.scaling * np.ravel(
                astra.data2d.get(self.proj_id)))
        elif geom.geom_type in self.type3d:
            return Rn(geom.proj_size).element(self.scaling * np.ravel(
                astra.data3d.get(self.proj_id)))
        else:
            raise Exception('Unknown geometry type.')

    def forward(self, rn_vector=None):
        """Forward projection."""

        # Store volume data of Rn vector in ASTRA memory
        if rn_vector is not None:
            self._store_volume(rn_vector)

        # Run algorithm
        astra.algorithm.run(self.fp_id)

        return self._get_projections()

    def _store_projections(self, rn_vector=Rn(1).element(1)):
        """Store projection data of Rn vector in ASTRA memory.

        Add description of order of dimensions.

        Parameters
        ----------
        :type rn_vector: odl.space.cartesian.Rn
        :param rn_vector: Vector in Rn containing 2D or 3D projection data.
        """

        geom = self.geom

        if geom.geom_type in self.type2d:
                astra.data2d.store(self.proj_id, rn_vector.data.reshape(
                    geom.angles.size, geom.det_col_count))
        elif geom.geom_type in self.type3d:
            astra.data3d.store(self.proj_id, rn_vector.data.reshape(
                geom.det_row_count, geom.angles.size, geom.det_col_count))
        else:
            raise Exception('Unknown geometry type.')

    def _get_volume(self):
        """Returns Rn vector containing the 2D or 3D volume data.

        Add description of order of dimensions.

        Returns
        -------
        :rtype: odl.space.cartesian.Rn
        :returns:  Vector in Rn containing 2D or 3D volume data.
        """

        geom = self.geom

        if geom.geom_type in self.type2d:
            return Rn(geom.vol_size).element(self.scaling * np.ravel(
                astra.data2d.get(self.vol_id)))
        elif geom.geom_type in self.type3d:
            return Rn(geom.vol_size).element(self.scaling * np.ravel(
                astra.data3d.get(self.vol_id)))
        else:
            raise Exception('Unknown geometry type.')

    def backward(self, rn_vector=None):
        """Back-projection."""

        # Store volume data of Rn vector in ASTRA memory
        if rn_vector is not None:
            self._store_projections(rn_vector)

        # Run algorithm
        astra.algorithm.run(self.bp_id)

        return self._get_volume()

    def fbp(self, rn_vector=None):
        """Back-projection."""

        # Store volume data of Rn vector in ASTRA memory
        if rn_vector is not None:
            self._store_projections(rn_vector)

        # Run algorithm
        astra.algorithm.run(self.fbp_id)

        return self._get_volume()


    def clear_astra_memory(self):
        """Clear memory allocated by ASTRA."""
        geom = self.geom

        if geom.vol_ndim == 2:
            astra.data2d.delete(self.vol_id)
        elif geom.vol_ndim == 3:
            astra.data3d.delete(self.vol_id)

        if geom.proj_ndim == 2:
            astra.data2d.delete(self.proj_id)
        elif geom.proj_ndim == 3:
            astra.data3d.delete(self.proj_id)

        astra.algorithm.delete(self.fp_id)
        astra.algorithm.delete(self.bp_id)
        if self.fbp_id is not None:
            astra.algorithm.delete(self.fbp_id)


class Geometry(object):

    def __init__(self,
                 scale_factor=1,
                 geometry_type='cone',
                 volume_shape=(60, 60, 60),
                 det_row_count=50, det_col_count=50,
                 angles=np.linspace(0, 2 * np.pi, 60, endpoint=False),
                 det_col_spacing=1.0, det_row_spacing=1.0,
                 source_origin=100.0, origin_detector=10.0,
                 voxel_size=1
                 ):
        self.geom_type = geometry_type
        self.vol_shape = [a * scale_factor for a in volume_shape]
        self.detector_spacing_x = det_col_spacing
        self.detector_spacing_y = det_row_spacing
        self.det_row_count = det_row_count * scale_factor
        self.det_col_count = det_col_count * scale_factor
        self.angles = angles
        self.source_origin = source_origin
        self.origin_detector = origin_detector
        # private
        self._voxel_size = 3 * (voxel_size,)
        self._full_angle_rad = np.abs((self.angles[-1] -
                                       self.angles[0]))

    @property
    def vol_size(self):
        return np.prod(self.vol_shape)

    @property
    def vol_ndim(self):
        return np.size(self.vol_shape)

    @property
    def proj_size(self):
        return self.det_row_count * self.det_col_count * np.size(self.angles)

    @property
    def proj_shape(self):
        if self.det_row_count == 1:
            return self.det_col_count, self.angles.size
        elif self.det_row_count > 1:
            return self.det_col_count, self.angles.size, self.det_row_count

    @property
    def voxel_size(self):
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, voxel_size_mm):
        self._voxel_size = voxel_size_mm

    @property
    def full_angle_rad(self):
        """Full angle of rotation in radians. Per default it takes the
        last and first entry of the angles_rad 1D-array. This values is
        uase to calculate the angular increment which is used in scaling
        factor for the projector forward and backward operators. Only
        meaningful for continous rotation with equidistant sampling points.
        """
        return self._full_angle_rad

    @full_angle_rad.setter
    def full_angle_rad(self, full_angular_range_in_rad):
        self._full_angle_rad = full_angular_range_in_rad


class ODLProjector(object):
    """ASTRA projector for ODL."""

    def __init__(self,
                 geometry_obj=Geometry(),
                 volume_space=Rn(Geometry().vol_size),
                 projections_space=Rn(Geometry().proj_size),
                 gpu_index=0):
        self.geom = geometry_obj
        self.vol_space = volume_space
        self.proj_space = projections_space
        self.gpu_index = gpu_index
        self.bp_id = None
        self.fp_id = None

        # Create volume geometry
        self.vol_geom = astra.create_vol_geom(self.geom.vol_shape)

        # Create projection geometry
        if self.geom.geom_type == 'cone':
            self.proj_geom = astra.create_proj_geom(
                self.geom.geom_type,
                self.geom.detector_spacing_x, self.geom.detector_spacing_y,
                self.geom.det_row_count, self.geom.det_col_count,
                self.geom.angles,
                self.geom.source_origin, self.geom.origin_detector)
        elif self.geom.geom_type == 'parallel':
            self.proj_geom = astra.create_proj_geom(
                'parallel', self.geom.detector_spacing_x,
                self.geom.det_col_count, self.geom.angles)

        # Allocate ASTRA memory for volume data and projection data
        if self.geom.vol_ndim == 2:
            self.volume_id = astra.data2d.create('-vol', self.vol_geom)
            self.proj_id = astra.data2d.create('-sino', self.proj_geom)
        elif self.geom.vol_ndim == 3:
            self.volume_id = astra.data3d.create('-vol', self.vol_geom)
            self.proj_id = astra.data3d.create('-sino', self.proj_geom)
        else:
            raise Exception("Invalid number of dimensions 'ndim'.")

        # self.scal_fac = self.geom.full_angle_rad / self.geom.angles.size
        # self.scal_fac = 1.0 / self.geom.angles.size
        self.scal_fac = self.geom.voxel_size[0] / self.geom.angles.size

    def forward(self, vol_space_vector):
        """Forward projection."""

        if self.geom.vol_ndim == 2:

            # Store volume data in ASTRA memory
            astra.data2d.store(self.volume_id,
                               vol_space_vector.data.reshape(
                                   self.geom.vol_shape))

            # Create algorithm object
            cfg = astra.astra_dict('FP_CUDA')
            cfg['option'] = {'GPUindex': self.gpu_index}
            cfg['ProjectionDataId'] = self.proj_id
            cfg['VolumeDataId'] = self.volume_id
            self.fp_id = astra.algorithm.create(cfg)

            # Run algorithm
            astra.algorithm.run(self.fp_id)

            # Retrieve projection data from ASTRA memory
            return self.proj_space.element(self.scal_fac * np.ravel(
                astra.data2d.get(self.proj_id)))

        elif self.geom.vol_ndim == 3:

            # Store volume data in ASTRA memory
            astra.data3d.store(self.volume_id,
                               vol_space_vector.data.reshape(
                                   self.geom.vol_shape))

            # Create algorithm object
            cfg = astra.astra_dict('FP3D_CUDA')
            cfg['option'] = {'GPUindex': self.gpu_index}
            cfg['ProjectionDataId'] = self.proj_id
            cfg['VolumeDataId'] = self.volume_id
            self.fp_id = astra.algorithm.create(cfg)

            # Run algorithm
            astra.algorithm.run(self.fp_id)

            # Retrieve projection data from ASTRA memory
            return self.proj_space.element(self.scal_fac * np.ravel(
                astra.data3d.get(self.proj_id)))

    def backward(self, proj_vector):
        """Backprojection."""

        if self.geom.vol_ndim == 2:

            # Store projection data in ASTRA memory
            astra.data2d.store(self.proj_id, proj_vector.data.reshape(
                self.geom.angles.size, self.geom.det_col_count))

            # Create algorithm object
            cfg = astra.astra_dict('BP_CUDA')
            cfg['option'] = {'GPUindex': self.gpu_index}
            cfg['ProjectionDataId'] = self.proj_id
            cfg['ReconstructionDataId'] = self.volume_id
            self.bp_id = astra.algorithm.create(cfg)

            # Run algorithms
            astra.algorithm.run(self.bp_id)

            # Retrieve projection from ASTRA memory
            return self.vol_space.element(self.scal_fac * np.ravel(
                astra.data2d.get(self.volume_id)))

        elif self.geom.vol_ndim == 3:

            # Store projection data in ASTRA memory
            astra.data3d.store(self.proj_id,
                               proj_vector.data.reshape(
                                   self.geom.det_row_count,
                                   self.geom.angles.size,
                                   self.geom.det_col_count))

            # Create algorithm object
            cfg = astra.astra_dict('BP3D_CUDA')
            cfg['option'] = {'GPUindex': self.gpu_index}
            cfg['ProjectionDataId'] = self.proj_id
            cfg['ReconstructionDataId'] = self.volume_id
            self.bp_id = astra.algorithm.create(cfg)

            # Run algorithms
            astra.algorithm.run(self.bp_id)

            # Retrieve projection from ASTRA memory
            return self.vol_space.element(self.scal_fac * np.ravel(
                astra.data3d.get(self.volume_id)))

    def clear_astra_memory(self):
        """Clear memory allocated by ASTRA."""

        if self.geom.vol_ndim == 2:
            astra.data2d.delete(self.volume_id)
            astra.data2d.delete(self.proj_id)
        elif self.geom.vol_ndim == 3:
            astra.data3d.delete(self.volume_id)
            astra.data3d.delete(self.proj_id)

        if self.fp_id is not None:
            astra.algorithm.delete(self.fp_id)
        if self.bp_id is not None:
            astra.algorithm.delete(self.bp_id)


class ODLProjectorOld(object):
    """ASTRA projector for ODL."""

    def __init__(self,
                 geometry_obj=Geometry(1),
                 vol_vector=None,
                 proj_vector=None, gpu_index=0):
        self.geom = geometry_obj
        if vol_vector is None:
            self.vol = Rn(self.geom.vol_size)
        else:
            self.vol = vol_vector
        if proj_vector is None:
            self.proj = Rn(self.geom.proj_size)
        else:
            self.proj = proj_vector
        self.gpu_index = gpu_index
        self.bp_id = None
        self.fp_id = None

        # Create volume geometry
        self.vol_geom = astra.create_vol_geom(self.geom.vol_shape)

        # Create projection geometry
        self.proj_geom = astra.create_proj_geom(
            self.geom.geom_type,
            self.geom.detector_spacing_x, self.geom.detector_spacing_y,
            self.geom.det_row_count, self.geom.det_col_count,
            self.geom.angles,
            self.geom.source_origin, self.geom.origin_detector)

        # Allocate ASTRA memory for volume data
        self.volume_id = astra.data3d.create('-vol', self.vol_geom)

        # Allocate ASTRA memory for projection data
        self.proj_id = astra.data3d.create('-sino', self.proj_geom)

    def forward(self, vol_vector=None):
        """Forward projection."""
        # Store volume data in ASTRA memory
        if vol_vector is None:
            astra.data3d.store(self.volume_id,
                               self.vol.data.reshape(self.geom.vol_shape))
        else:
            astra.data3d.store(self.volume_id,
                               vol_vector.data.reshape(self.geom.vol_shape))

        # Create algorithm object
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = self.proj_id
        cfg['VolumeDataId'] = self.volume_id
        self.fp_id = astra.algorithm.create(cfg)

        # Run algorithm
        astra.algorithm.run(self.fp_id)

        # Retrieve projection data from ASTRA memory
        self.proj.data[:] = np.ravel(astra.data3d.get(self.proj_id))

    def backward(self, proj_vector=None):
        """Backprojection."""

        # Store projection data in ASTRA memory
        if proj_vector is None:
            astra.data3d.store(self.proj_id,
                               self.proj.data.reshape(
                                   self.geom.det_row_count,
                                   self.geom.angles.size,
                                   self.geom.det_col_count))
        else:
            astra.data3d.store(self.proj_id,
                               proj_vector.data.reshape(
                                   self.geom.det_row_count,
                                   self.geom.angles.size,
                                   self.geom.det_col_count))

        # Create algorithm object
        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = self.proj_id
        cfg['ReconstructionDataId'] = self.volume_id
        self.bp_id = astra.algorithm.create(cfg)

        # Run algorithms
        astra.algorithm.run(self.bp_id)

        # Retrieve projection from ASTRA memory
        self.vol.data[:] = np.ravel(astra.data3d.get(self.volume_id))

    def clear_astra_memory(self):
        """Clear memory allocated by ASTRA."""

        astra.data3d.delete(self.volume_id)
        astra.data3d.delete(self.proj_id)
        if self.fp_id is not None:
            astra.algorithm.delete(self.fp_id)
        if self.bp_id is not None:
            astra.algorithm.delete(self.bp_id)


class Projector(object):
    """ASTRA projector interface.

    :type projection_data: int | ndarray
    :type volume_data: int | ndarray
    """

    def __init__(self, geometry_type='cone',
                 num_voxel=(100, 100, 100),
                 det_row_count=100, det_col_count=100,
                 angles=np.linspace(0, 2 * np.pi, 180, endpoint=False),
                 det_col_spacing=1.0, det_row_spacing=1.0,
                 source_origin=100.0, origin_detector=10.0,
                 volume_data=1, projection_data=1,
                 gpu_index=0):
        self.geometry_type = geometry_type
        self.num_voxel = num_voxel
        self.detector_spacing_x = det_col_spacing
        self.detector_spacing_y = det_row_spacing
        self.det_row_count = det_row_count
        self.det_col_count = det_col_count
        self.angles = angles
        self.source_origin = source_origin
        self.origin_detector = origin_detector
        self.volume_data = volume_data
        self.projection_data = projection_data
        self.gpu_index = gpu_index

        # Create volume geometry
        self.volume_geom = astra.create_vol_geom(self.num_voxel)

        # Create projection geometry
        self.projection_geom = astra.create_proj_geom(
            self.geometry_type,
            self.detector_spacing_x, self.detector_spacing_y,
            self.det_row_count, self.det_col_count,
            self.angles,
            self.source_origin, self.origin_detector)

        # Allocate and store volume data in ASTRA memory
        self.volume_id = astra.data3d.create(
            '-vol',
            self.volume_geom,
            self.volume_data)

        # Allocate and store projection data in ASTRA memory
        self.projection_id = astra.data3d.create(
            '-sino',
            self.projection_geom,
            self.projection_data)

        # Create algorithm object: forward projector
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = self.projection_id
        cfg['VolumeDataId'] = self.volume_id
        self.forward_alg_id = astra.algorithm.create(cfg)

        # Create algorithm object: backward projector
        cfg = astra.astra_dict('BP3D_CUDA')
        # classmethod?
        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = self.projection_id
        cfg['ReconstructionDataId'] = self.volume_id
        self.backward_alg_id = astra.algorithm.create(cfg)

    # @jit
    def set_volume_data(self, volume_data):
        """Store volume in ASTRA memory."""
        astra.data3d.store(self.volume_id, volume_data)

    def set_projection_data(self, projection_data):
        """Store projection data in ASTRA memory."""
        astra.data3d.store(self.projection_id, projection_data)

    def get_volume_data(self):
        """Retrieve volume data from ASTRA memory."""
        return astra.data3d.get(self.volume_id)

    def get_projection_data(self):
        """Retrieve projection data from ASTRA memory."""
        return astra.data3d.get(self.projection_id)

    def forward(self):
        """Run forward projection."""
        astra.algorithm.run(self.forward_alg_id)
        self.projection_data = astra.data3d.get(self.projection_id)

    def backward(self):
        """Run backward projection."""
        astra.algorithm.run(self.backward_alg_id)
        self.volume_data = astra.data3d.get(self.volume_id)

    def clear(self):
        """Clear internal ASTRA memory."""
        astra.data3d.delete(self.volume_id)
        astra.data3d.delete(self.projection_id)
        astra.algorithm.delete(self.forward_alg_id)
        astra.algorithm.delete(self.backward_alg_id)

    # @property
    # def volume_data(self):
    #     """Numpy array of volume data"""
    # Add attribute as a property
    # volume_data = property(get_volume_data, set_volume_data)

    # Add attribute as a property
    # projection_data = property(get_projection_data, set_projection_data)

    @property
    def projection_shape(self):
        """Return shape of ndarray."""
        return np.shape(self.projection_data)

    @property
    def volume_shape(self):
        """Return shape of ndarray."""
        return np.shape(self.volume_data)


class ForwardProjector(Operator):
    """Create forward projector using the ASTRA Toolbox via its PyASTRA interface.

    Attributes
    ----------
    """

    def __init__(self, geometry_type='cone',
                 num_voxel=(100, 100, 100),
                 det_row_count=100, det_col_count=100,
                 angles=np.linspace(0, 2 * np.pi, 180, endpoint=False),
                 det_col_spacing=1.0, det_row_spacing=1.0,
                 source_origin=100.0, origin_detector=10.0,
                 alg_string='FP3D_CUDA', gpu_index=0):
        self.geometry_type = geometry_type
        self.num_voxel = num_voxel
        self._domain = Rn(np.prod(num_voxel))
        self._range = Rn(det_col_count * det_row_count * np.size(angles))
        self.detector_spacing_x = det_col_spacing
        self.detector_spacing_y = det_row_spacing
        self.det_row_count = det_row_count
        self.det_col_count = det_col_count
        self.angles = angles
        self.source_origin = source_origin
        self.origin_detector = origin_detector
        self.alg_string = alg_string
        self.gpu_index = gpu_index
        self._adjoint = BackwardProjector(
            geometry_type=geometry_type,
            num_voxel=num_voxel,
            det_row_count=det_row_count, det_col_count=det_col_count,
            angles=angles,
            det_col_spacing=det_col_spacing, det_row_spacing=det_row_spacing,
            source_origin=source_origin, origin_detector=origin_detector,
            gpu_index=gpu_index)

    def _apply(self, volume, projections):
        """Apply forward projector.

        Parameters
        ----------

        volume_data : RNVector
              RNVector of dimension self.domain.n which contains the data be
              projected.

        projections : RNVector
              RNVector of dimension self.range.n the projections are written
              to.


        Examples
        --------
        >>> num_voxel = (100, 100, 100)

        #>>> phantom = Cuboid(shape=num_voxel).data
        >>> phantom = np.zeros(num_voxel)

        # <class 'RL.space.euclidean.RN'>
        >>> rn = Rn(phantom.size)

        # <class 'RL.space.euclidean.Vector'>
        >>> rn_phantom = rn.element(phantom.flatten())

        # <class 'astra.ForwardProjector'>
        >>> fp = ForwardProjector(num_voxel=num_voxel)

        # <class 'RL.space.euclidean.Vector'>
        >>> rn_proj = fp(rn_phantom)

        >>> proj = np.reshape(rn_proj.data, (100, 180, 100))
        >>> proj.shape
        (100, 180, 100)
        """

        # Create volume geometry
        vol_geom = astra.create_vol_geom(self.num_voxel)

        # Create projection geometry
        proj_geom = astra.create_proj_geom(
            self.geometry_type,
            self.detector_spacing_x, self.detector_spacing_y,
            self.det_row_count, self.det_col_count,
            self.angles,
            self.source_origin, self.origin_detector)

        # Allocate and store volume data in ASTRA memory
        volume_id = astra.data3d.create(
            '-vol',
            vol_geom,
            volume.data.reshape(self.num_voxel))

        # Allocate ASTRA memeory for projection data
        proj_id = astra.data3d.create('-sino', proj_geom, 0)

        # Create algorithm object
        cfg = astra.astra_dict(self.alg_string)
        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = proj_id
        cfg['VolumeDataId'] = volume_id
        alg_id = astra.algorithm.create(cfg)

        # Run algorithms
        # with(Timer("Projection")):
        astra.algorithm.run(alg_id)

        # Retrieve projection from ASTRA memory
        projections.data[:] = np.ravel(astra.data3d.get(proj_id))

        # Free ASTRA memory
        astra.data3d.delete(volume_id)
        astra.data3d.delete(proj_id)
        astra.algorithm.delete(alg_id)

    @property
    def adjoint(self):
        return self._adjoint

    @property
    def domain(self):
        return self._domain

    @property
    def range(self):
        return self._range


class BackwardProjector(Operator):
    """Create back projector using the ASTRA Toolbox via its PyASTRA interface.

    Attributes
    ----------
    """

    def __init__(self, geometry_type='cone',
                 num_voxel=(100, 100, 100),
                 det_row_count=100, det_col_count=100,
                 angles=np.linspace(0, 2 * np.pi, 180, endpoint=False),
                 det_col_spacing=1.0, det_row_spacing=1.0,
                 source_origin=100.0, origin_detector=10.0,
                 alg_string='BP3D_CUDA', gpu_index=0):
        self.geometry_type = geometry_type
        self.num_voxel = num_voxel
        self._domain = Rn(det_col_count * det_row_count * np.size(angles))
        self._range = Rn(np.prod(num_voxel))
        self.detector_spacing_x = det_col_spacing
        self.detector_spacing_y = det_row_spacing
        self.det_row_count = det_row_count
        self.det_col_count = det_col_count
        self.angles = angles
        self.source_origin = source_origin
        self.origin_detector = origin_detector
        self.alg_string = alg_string
        self.gpu_index = gpu_index

    def _apply(self, projections, volume):
        """ Apply back projector.

        Parameters
        ----------

        projections : RNVector

              RNVector of dimension self.range.n which contains the
              projection data (sinogram).

        volume_data : RNVector

              RNVector of dimension self.domain.n onto which the projection
              will be projected back.


        Examples
        --------
        >>> num_voxel = (100, 100, 100)
        >>> det_row_count = 100
        >>> det_col_count = 100
        >>> angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        >>> phantom = Cuboid(shape=num_voxel).data

        # <class 'RL.space.euclidean.RN'>
        >>> rn = Rn(phantom.size)

        # <class 'RL.space.euclidean.Vector'>
        >>> rn_phantom = rn.element(phantom.ravel())

        # <class 'astra.ForwardProjector'>
        >>> fp = ForwardProjector(num_voxel=num_voxel, \
        det_row_count=det_row_count, det_col_count=det_col_count, \
        angles=angles)

        # <class 'RL.space.euclidean.Vector'>
        >>> rn_proj = fp(rn_phantom)

        >>> bp = BackwardProjector(num_voxel=num_voxel,\
        det_row_count=det_row_count, det_col_count=det_col_count, \
        angles=angles)

        >>> type(bp)
        <class 'pyastra.BackwardProjector'>

        # <class 'RL.space.euclidean.Vector'>
        >>> rn_rec = bp(rn_proj)

        >>> rec = np.reshape(rn_rec.data, num_voxel)
        >>> rec.shape
        (100, 100, 100)
        """

        # Create volume geometry
        vol_geom = astra.create_vol_geom(self.num_voxel)

        # Create projection geometry
        proj_geom = astra.create_proj_geom(
            self.geometry_type,
            self.detector_spacing_x, self.detector_spacing_y,
            self.det_row_count, self.det_col_count,
            self.angles,
            self.source_origin, self.origin_detector)

        # Allocate ASTRA memeory and store projection data in it
        proj_id = astra.data3d.create(
            '-sino',
            proj_geom,
            projections.data.reshape(
                self.det_row_count,
                self.angles.size,
                self.det_col_count))

        # proj_data = np.reshape(
        #     np.ravel(projections.data),
        #     (self.det_row_count, self.angles.size, self.det_col_count))
        # proj_id = astra.data3d.create('-sino', proj_geom, proj_data)

        # Allocate ASTRA memory for volume data
        volume_id = astra.data3d.create('-vol', vol_geom, 0)

        # Create algorithm object
        cfg = astra.astra_dict(self.alg_string)
        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = proj_id
        cfg['ReconstructionDataId'] = volume_id
        alg_id = astra.algorithm.create(cfg)

        # Run algorithms
        # with(Timer("Projection")):
        astra.algorithm.run(alg_id)

        # Retrieve projection from ASTRA memory
        volume.data[:] = np.ravel(astra.data3d.get(volume_id))

        # Free ASTRA memory
        astra.data3d.delete(volume_id)
        astra.data3d.delete(proj_id)
        astra.algorithm.delete(alg_id)

    @property
    def domain(self):
        return self._domain

    @property
    def range(self):
        return self._range

    @property
    def adjoint(self):
        return 0
