#! /usr/bin/env python
"""
Module for import of CT data stored in a Matlab file.

"""

from __future__ import division
import numpy as np
import os
import scipy.io as sio
from numba import jit
# from pyastra import Geometry

# __metaclass__ = type


class Data(object):
    """Class to specify parameters for CT data sets and to provide modules for
    reading of data, retrieving data attributes (shape, roi, etc),
    preprocessing (remove infs, nans, etc).
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        # Data file parameters
        self._filename = ''
        self._parent_path = os.getenv('HOME') + '/data/gate/'
        self._data_format = 'mat'
        self._absolute_filename = None
        # Data structure parameters
        self._projections = np.array([])
        self._matlab_fieldname = 'detector'
        self._permute_order = None
        self._shape = ()
        self.astype = 'float32'
        # Geometric paramters
        self._angles_rad = np.array([])
        self._angle_range_rad = None
        self._geometry_type = 'cone'
        self._dist_source_origin = None
        self._dist_origin_det = None
        self._dist_source_det = None
        self._det_width_mm = None
        self._roi_cubic_width_mm = None
        # Data processing parameters
        self.remove_infs = False
        self.remove_nans = False
        self.normalize = False
        self.take_neg_log = False

    # Name of data set (filename)
    @property
    def filename(self):
        """File name of the data set without data format sufffix."""
        return self._filename

    @filename.setter
    def filename(self, filename='File name of data set without suffix'):
        """Set file name (without suffix) to 'name'."""
        self._filename = filename
        self._absolute_filename = self._parent_path + self._filename \
            + self._data_format

    # Path containing data set file
    @property
    def parent_path(self):
        """Absolute path to the folder containing the projection data set.
        Default: ~/data/gate/"""
        return self._parent_path

    @parent_path.setter
    def parent_path(self, parent_path):
        """Set parent path."""
        self._parent_path = parent_path
        self._absolute_filename = self._parent_path + self._filename + \
            self._data_format

    # Format of data set
    @property
    def data_format(self):
        """File format of projection data. So far only MATLAB files support."""
        return self._data_format

    @data_format.setter
    def data_format(self, data_format):
        """Set data format."""
        if data_format[0] == '.':
            self._data_format = data_format
        else:
            self._data_format = '.' + data_format
        self._absolute_filename = self._parent_path + self._filename + \
            self._data_format

    # Absolute filename
    @property
    def absolute_filename(self):
        """Define absolute filename instead of using 'parent_path', 'filename',
         and 'data_set_format'."""
        return self._absolute_filename

    @absolute_filename.setter
    def absolute_filename(self, absolute_filename):
        """Set absolute filename."""
        self._absolute_filename = absolute_filename
        _root, self._data_format = os.path.splitext(absolute_filename)
        self._parent_path = os.path.dirname(absolute_filename)
        self._filename = os.path.basename(_root)

    # Fieldname within MATLAB workspace
    @property
    def matlab_fieldname(self):
        """Fieldname of data set within MATLAB workspace which is required in
        order to import it as numpy array."""
        return self._matlab_fieldname

    @matlab_fieldname.setter
    def matlab_fieldname(self, fieldname_in_matlab_workspace):
        """Set name of field the data has in MATLAB Workspace."""
        assert isinstance(fieldname_in_matlab_workspace, str)
        self._matlab_fieldname = fieldname_in_matlab_workspace

    # Permutation order
    @property
    def permute_order(self):
        """Defines the order of permutation of the dimensions of the data
        volume in order to coincide with PyASTRA notation. Default is not
        permutation."""
        return self._permute_order

    @permute_order.setter
    def permute_order(self, order_of_permutation):
        """Set permutation order."""
        self._permute_order = order_of_permutation

    # Type of geometry
    @property
    def geometry_type(self):
        """Define data acquistion geometry."""
        return self._geometry_type

    @geometry_type.setter
    def geometry_type(self, geometry_type):
        """Set data acquistion geometry."""
        self._geometry_type = geometry_type

    # # Geometry object
    # @property
    # def geometry(self):
    #     """Create geometry object to be pass through to ASTRA projector
    #     instance for ODL."""
    #     return Geometry(geometry_type=self._geometry_type,
    #                     volume_shape=self._shape,
    #                     det_row_count=0,
    #                     det_col_count=0,
    #                     angles=self.angles_rad,
    #                     det_col_spacing=1.0,
    #                     det_row_spacing=1.0,
    #                     source_origin=self.distance_source_origin_mm,
    #                     origin_detector=self.distance_origin_detector_mm
    #                     )

    # Angles in radiants
    @property
    def angles_rad(self):
        """Discrete angles of rotation in radiants.

        Returns
        -------
        1D numpy array
        """
        return self._angles_rad

    @angles_rad.setter
    def angles_rad(self, angles_rad):
        """Set set of discrete angles."""
        self._angles_rad = np.array(angles_rad)

    def __set_angles_rad(self):
        """Helper functions to set angle set."""
        try:
            self._angles_rad = self._angle_range_rad * np.arange(
                self._shape[1]) / self._shape[1]
        except IndexError:
            pass

    # Full angle range of rotation
    @property
    def angle_range_rad(self):
        """Full angle range of rotation."""
        return self._angle_range_rad

    @angle_range_rad.setter
    def angle_range_rad(self, full_angle_range_rad):
        """Set angle of rotation."""
        self._angle_range_rad = float(full_angle_range_rad)
        if self._angles_rad.shape[0] == 0:
            self.__set_angles_rad()

    @property
    def shape(self):
        """Shape of projection data."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        """Set shape."""
        self.shape = shape

    # Distances
    @property
    def distance_source_origin_mm(self):
        """Distance between source and origin of reconstruction volume."""
        return self._dist_source_origin

    @distance_source_origin_mm.setter
    def distance_source_origin_mm(self, dist_source_origin):
        """Set distance between source and origin."""
        self._dist_source_origin = float(dist_source_origin)
        try:
            self._dist_origin_det = self._dist_source_det - dist_source_origin
        except (TypeError, AttributeError):
            pass
        try:
            self._dist_source_det = self._dist_origin_det + dist_source_origin
        except (TypeError, AttributeError):
            pass

    @property
    def distance_origin_detector_mm(self):
        """Distance between origin of reconstruction volume and detector."""
        return self._dist_origin_det

    @distance_origin_detector_mm.setter
    def distance_origin_detector_mm(self, dist_origin_det):
        """Set distance between origin of reconstruction volume and
        detector."""
        self._dist_origin_det = float(dist_origin_det)
        try:
            self._dist_source_origin = self._dist_source_det - dist_origin_det
        except (TypeError, AttributeError):
            pass
        try:
            self._dist_source_det = self._dist_source_origin + dist_origin_det
        except (TypeError, AttributeError):
            pass

    @property
    def distance_source_detector_mm(self):
        """Distance between origin of reconstruction volume and detector."""
        return self._dist_source_det

    @distance_source_detector_mm.setter
    def distance_source_detector_mm(self, dist_source_det):
        """Set istance between origin of reconstruction volume and detector."""
        self._dist_source_det = float(dist_source_det)
        try:
            self._dist_source_origin = - self._dist_origin_det + \
                dist_source_det
        except (TypeError, AttributeError):
            pass
        try:
            self._dist_origin_det = - self._dist_source_origin + \
                dist_source_det
        except (TypeError, AttributeError):
            pass

    @property
    def detector_width_mm(self):
        """Full width of detector in millimeter."""
        return self._det_width_mm

    @detector_width_mm.setter
    def detector_width_mm(self, detector_width_mm):
        """Set full width of detector in millimeter."""
        self._det_width_mm = float(detector_width_mm)

    @property
    def roi_cubic_width_mm(self):
        """Cuboidal region of interest (ROI) in millimeter."""
        self._compute_roi()
        return self._roi_cubic_width_mm

    def _compute_roi(self):
        """Helper function to compute cuboidal region of interest (ROI) in
        millimeter."""
        try:
            pdet_width_mm = self._dist_source_origin / self._dist_source_det *\
                self._det_width_mm
            self._roi_cubic_width_mm = np.sqrt(2) * pdet_width_mm / 2 * \
                np.sqrt(1 - (pdet_width_mm / self._dist_source_origin / 2) ** 2
                        )
        except TypeError:
            raise TypeError(
                "Not all parameters were set in order to compute roi: "
                "{0}, {1}, {2}".format(
                    self._dist_source_origin,
                    self._dist_source_det,
                    self._det_width_mm))

    @property
    def projections(self):
        """Numpy array of projection data."""
        # if self._projections.size == 0:
        #     self.__load()
        return self._projections

    @projections.setter
    def projections(self, ndarray_of_projections=np.array([])):
        """Set numpy array of projection data.

        :type ndarray_of_projections: numpy.ndarray
        :param ndarray_of_projections: numpy.ndarray containing the
        projection data.
        """
        self._projections = ndarray_of_projections

    @property
    def dtype(self):
        """Data type."""
        return self.projections.dtype

    def load(self):
        """Explicitly load projections instead of accessing them via the
        property attribute self.projections. This is in order to make
        property accessible which need the projections to be loaded."""
        self.__load()

    @jit
    def __load(self):
        """Private function which loads the data into memory."""

        assert os.path.isfile(
            self._absolute_filename), "File does not exist:" + \
                                      self._absolute_filename
        assert isinstance(self._absolute_filename, str)

        # Read data
        if self._permute_order is not None:
            self._projections = np.transpose(
                sio.loadmat(
                    self._absolute_filename,
                    variable_names=self._matlab_fieldname)[
                    self._matlab_fieldname],
                self._permute_order
            ).astype(self.astype)

        else:
            self._projections = sio.loadmat(
                self._absolute_filename,
                variable_names=self._matlab_fieldname)[
                self._matlab_fieldname].astype(self.astype)

        # Postprocess data
        self._shape = np.shape(self._projections)
        self.__set_angles_rad()
        self.raw_data_min = self._projections.min()
        self.raw_data_max = self._projections.max()
        self.raw_data_mean = self._projections.mean()
        if self.remove_infs:
            self._remove_infs()
        if self.remove_nans:
            self._remove_nans()
        if self.normalize:
            self._normalize()
        if self.take_neg_log:
            self._take_neg_log()

    @jit
    def _normalize(self):
        """Normalize data."""
        assert np.all(self.projections >= 0), "Negative values in " \
            "projection data."
        self._projections /= self.normalize - 1 + self.projections.max()

    @jit
    def _take_neg_log(self):
        """Take the negative natural logarithm of the projection data."""
        assert np.all(self.projections > 0), "Zero values in projection data"
        self._projections = - np.log(self._projections)

    @jit
    def _remove_infs(self):
        """Replace inf values by 0 from data set."""
        self._projections[np.isinf(self._projections)] = 0

    @jit
    def _remove_nans(self):
        """Replace nan values by 0 from data set."""
        self._projections[np.isnan(self._projections)] = 0


# Create dictionary of data sets. Dictionary values are instances of
# <class 'Data'>. By now the dictionary consists mainly of GATE simulated data
# for conebeam geometry.
sets = dict()

# ASTRA geometries

KEYWORD = 'parallel'
DATA = Data()
DATA.parent_path = '/home/jmoosmann/data/matlab/'
DATA.filename = 'astra_geometries'
DATA.data_format = 'mat'
# DATA.permute_order = (0, 1)
DATA.matlab_fieldname = KEYWORD
DATA.angle_range_rad = 2 * np.pi
DATA.geometry_type = KEYWORD
DATA.detector_width_mm = 100
sets[KEYWORD] = DATA

KEYWORD = 'fanflat'
DATA = Data()
DATA.parent_path = '/home/jmoosmann/data/matlab/'
DATA.filename = 'astra_geometries'
DATA.data_format = 'mat'
# DATA.permute_order = (1, 0)
DATA.matlab_fieldname = KEYWORD
DATA.angle_range_rad = 2 * np.pi
DATA.geometry_type = KEYWORD
DATA.detector_width_mm = 100
DATA.distance_source_origin_mm = 80
DATA.distance_origin_detector_mm = 20
sets[KEYWORD] = DATA

KEYWORD = 'parallel3d'
DATA = Data()
DATA.parent_path = '/home/jmoosmann/data/matlab/'
DATA.filename = 'astra_geometries'
DATA.data_format = 'mat'
# DATA.permute_order = (2, 1, 0)
DATA.matlab_fieldname = KEYWORD
DATA.angle_range_rad = 2 * np.pi
DATA.geometry_type = KEYWORD
DATA.detector_width_mm = 100
sets[KEYWORD] = DATA

KEYWORD = 'cone'
DATA = Data()
DATA.parent_path = '/home/jmoosmann/data/matlab/'
DATA.filename = 'astra_geometries'
DATA.data_format = 'mat'
DATA.matlab_fieldname = KEYWORD
DATA.angle_range_rad = 2 * np.pi
DATA.geometry_type = KEYWORD
DATA.detector_width_mm = 100
DATA.distance_source_origin_mm = 80
DATA.distance_origin_detector_mm = 20
sets[KEYWORD] = DATA


# GATE data

KEYWORD = 1
DATA = Data()
DATA.filename = 'detector_two_spheres_Astra_20150313'
DATA.matlab_fieldname = 'detector_astra'
DATA.permute_order = (2, 1, 0)
DATA.angle_range_rad = -2 * np.pi
DATA.geometry_type = 'cone'
DATA.distance_source_origin_mm = 280
DATA.distance_origin_detector_mm = 20
DATA.detector_width_mm = 50
sets[KEYWORD] = DATA

KEYWORD = 2
DATA = Data()
DATA.filename = '20150317_water_spheres_all_photons'
DATA.matlab_fieldname = 'detector_astra'
DATA.permute_order = (0, 1, 2)
DATA.angle_range_rad = -2 * np.pi
DATA.geometry_type = 'cone'
DATA.distance_source_origin_mm = 280
DATA.distance_origin_detector_mm = 20
DATA.detector_width_mm = 50
sets[KEYWORD] = DATA

KEYWORD = 3
DATA = Data()
DATA.filename = '20150317_water_spheres_all_photons'
DATA.matlab_fieldname = 'detector_astra'
DATA.permute_order = (0, 1, 2)
DATA.angle_range_rad = -2 * np.pi
DATA.geometry_type = 'cone'
DATA.distance_source_origin_mm = 280
DATA.distance_origin_detector_mm = 20
DATA.detector_width_mm = 50
sets[KEYWORD] = DATA

KEYWORD = 9
DATA = Data()
DATA.filename = '20150410_water_spheres_high_act'
DATA.matlab_fieldname = 'detector_astra'
DATA.permute_order = (2, 1, 0)
DATA.angle_range_rad = -2 * np.pi
DATA.geometry_type = 'cone'
DATA.distance_source_origin_mm = 190
DATA.distance_origin_detector_mm = 110
DATA.detector_width_mm = 100
DATA.take_neg_log = 1
sets[KEYWORD] = DATA

KEYWORD = 12
DATA = Data()
DATA.filename = '20150528_CBCT_skull'
DATA.data_format = 'mat'
DATA.matlab_fieldname = 'detector_astra_full'
DATA.permute_order = (2, 1, 0)
DATA.angle_range_rad = 2 * np.pi
DATA.geometry_type = 'cone'
DATA.distance_source_detector_mm = 1085.6
DATA.distance_origin_detector_mm = 490.6
DATA.detector_width_mm = 500
DATA.remove_infs = True
DATA.remove_infs = True
sets[KEYWORD] = DATA

KEYWORD = 13
DATA = Data()
DATA.filename = '20150615_microCT_phantom_test'
DATA.data_format = 'mat'
DATA.matlab_fieldname = 'proj'
DATA.permute_order = (1, 2, 0)
DATA.distance_source_detector_mm = 366
DATA.distance_source_origin_mm = 270
DATA.detector_width_mm = 120
DATA.angle_range_rad = - 2 * np.pi
DATA.normalize = True
DATA.take_neg_log = True
sets[KEYWORD] = DATA

KEYWORD = 14
DATA = Data()
DATA.filename = '20150616_CT_for_SPECT_attenuation'
DATA.data_format = 'mat'
DATA.matlab_fieldname = 'detector_astra_projection'
DATA.permute_order = (2, 1, 0)
DATA.angle_range_rad = - 2 * np.pi
DATA.geometry_type = 'cone'
DATA.distance_source_detector_mm = 1085.6
DATA.distance_origin_detector_mm = 490.6
DATA.detector_width_mm = 950
DATA.normalize = True
DATA.take_neg_log = True
sets[KEYWORD] = DATA

KEYWORD = 15
DATA = Data()
DATA.filename = 'Sinogram_greyscale_nsdv1procent'
DATA.data_format = 'mat'
DATA.matlab_fieldname = 'sinogram'
DATA.permute_order = (2, 1, 0)
DATA.angle_range_rad = - 2 * np.pi
DATA.detector_width_mm = 950
DATA.geometry_type = 'parallel3d'
DATA.normalize = True
DATA.take_neg_log = True
sets[KEYWORD] = DATA

if __name__ == "__main__":
    import doctest
    doctest.testmod()
