"""Unit test for CT data module."""

import unittest
from ctdata import Data
from ctdata import sets as data_sets


class DataTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_instance(self):
        d = Data()
        self.assertEqual(Data, d.__class__)

    def test_set_distances(self):
        """After setting two distances, the third should have been set
        automatically. """

        d = Data()
        self.assertFalse(d.distance_source_origin_mm)
        self.assertFalse(d.distance_origin_detector_mm)
        self.assertFalse(d.distance_source_detector_mm)

        d.distance_source_origin_mm = 1
        self.assertEqual(1, d.distance_source_origin_mm)
        self.assertFalse(d.distance_origin_detector_mm)
        self.assertFalse(d.distance_source_detector_mm)

        d.distance_origin_detector_mm = 2
        self.assertEqual(1, d.distance_source_origin_mm)
        self.assertEqual(2, d.distance_origin_detector_mm)
        self.assertEqual(3, d.distance_source_detector_mm)

    def test_compute_roi(self):
        d = Data()
        try:
            d.roi_cubic_width_mm
        except TypeError, e:
            self.assertEqual('Not all parameters were set in order to '
                             'compute roi: None, None, None', e[0])
        else:
            raise

        d.distance_source_detector_mm = 100
        d.distance_source_origin_mm = 60
        dist = (d.distance_source_origin_mm, d.distance_origin_detector_mm,
                d.distance_source_detector_mm)

        self.assertEqual((60, 40, 100), dist)

        d.detector_width_mm = 100
        self.assertAlmostEqual(36.7423461417, d.roi_cubic_width_mm)

    def test_data_set(self):

        ds = data_sets[13]
        self.assertEqual(Data, ds.__class__)
        ds.load()
        proj = ds.projections
        self.assertEqual('ndarray', proj.__class__.__name__)

        self.assertEqual((240, 180, 240), proj.shape)

    def test_creation_of_geometry_object(self):

        ds = data_sets[14]
        self.assertTrue(ds.geometry)

    def test_data_postprocessing(self):

        d = data_sets[14]
        d.normalize = False
        d.take_neg_log = False
        self.assertFalse(d.normalize)
        self.assertFalse(d.take_neg_log)
        d.load()
        print 'no post-processing: ', d.projections.min(), d.projections.max()
        d._normalize()
        print 'normalized manyally:', d.projections.min(), d.projections.max()
        d.normalize = True
        d.load()
        print 'normalized via flag:', d.projections.min(), d.projections.max()
        self.assertEqual(d.projections.max(), 1)
        d._take_neg_log()
        print 'normalized + neg log:', d.projections.min(), d.projections.max()


if __name__ == '__main__':
    unittest.main()
