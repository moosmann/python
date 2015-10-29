#! /usr/bin/env python
"""
Module for import of CT data stored in a Matlab file.

"""

import numpy as np
import os
import scipy.io

__metaclass__ = type

class Data:

    # def __init__(self, folder):
    #     self.folderName = folder

    name = 'Name_of_data_file_without_suffix'
    def set_name(self, name):
        self.name = name
        self.filename = self.parent_path + name + self.data_format

    data_format = '.mat'
    def set_data_format(self, data_format):
        self.data_format = data_format
        self.filename = self.parent_path + self.name +  data_format

    parent_path = os.getenv('HOME') + '/data/gate/'
    def set_parent_path(self, parent_path):
        self.parent_path = parent_path
        self.filename = parent_path + self.name + self.data_format

    fieldname = 'detector_astra'
    def set_fieldname(self, fieldname):
        self.fieldname = fieldname

    filename = parent_path + name + data_format
    def set_filename(self, filename):
        self.filename = filename
        root, self.data_format = os.path.splitext(filename)
        self.parent_path = os.path.dirname(filename)
        self.name = os.path.basename(root)

    angle_range__rad = -2 * np.pi
    def set_angle_range__rad(self, angle_range__rad):
        self.angle_range__rad = float(angle_range__rad)
        if self.angles__rad.any():
            self.angles__rad = self.angle_range__rad * np.arange(self.shape[1]) / self.shape[1]

    angles__rad = np.array([])

    # Distances between source, origin, and detector
    distance_source_origin__mm = 0
    def set_distance_source_origin__mm(self, distance_source_origin__mm):
        self.distance_source_origin__mm = float(distance_source_origin__mm)
        if not self.distance_source_detector__mm:
            self.distance_source_detector__mm = distance_source_origin__mm + self.distance_origin_detector__mm
        if not self.distance_origin_detector__mm:
            self.distance_origin_detector__mm = self.distance_source_detector__mm - distance_source_origin__mm

    distance_origin_detector__mm = 0
    def set_distance_origin_detector__mm(self, distance_origin_detector__mm):
        self.distance_origin_detector__mm = float(distance_origin_detector__mm)
        if not self.distance_source_detector__mm:
            self.distance_source_detector__mm = self.distance_source_origin__mm + distance_origin_detector__mm
        if not self.distance_source_origin__mm:
            self.distance_source_origin__mm = self.distance_source_detector__mm - distance_origin_detector__mm

    distance_source_detector__mm = 0
    def set_distance_source_detector__mm(self, distance_source_detector__mm):
        self.distance_source_detector__mm = float(distance_source_detector__mm)
        if not self.distance_origin_detector__mm:
            self.distance_origin_detector__mm = distance_source_detector__mm - self.distance_source_origin__mm
        if not self.distance_source_origin__mm:
            self.distance_source_origin__mm = distance_source_detector__mm - self.distance_origin_detector__mm


    detectorWidth_mm = 100
    def set_detector_width__mm(self, detector_width__mm):
        self.detector_width__mm = float(detector_width__mm)

    permute_order = (0, 1, 2)
    def set_permute_order(self, permute_order):
        self.permute_order = permute_order

    def load(self):
        self.projections = np.transpose(scipy.io.loadmat(self.filename)[self.fieldname], self.permute_order)
        #self.projections = scipy.io.loadmat(self.filename)[self.fieldname]
        self.shape = self.projections.shape
        self.angles__rad = self.angle_range__rad * np.arange(self.shape[1]) / self.shape[1]
        return self.projections
        
        


d1 = Data()
d1.set_name('20150528_CBCT_skull')
d1.set_fieldname('detector_astra_full')
d1.set_angle_range__rad(-2*np.pi)
d1.set_distance_source_detector__mm(1085.6)
d1.set_distance_origin_detector__mm(490.6)
d1.load()

if 0:
    print d1.distance_source_origin__mm, d1.distance_origin_detector__mm, d1.distance_source_detector__mm
    d1.set_distance_source_detector__mm(1085.6)
    print d1.distance_source_origin__mm, d1.distance_origin_detector__mm, d1.distance_source_detector__mm
    d1.set_distance_origin_detector__mm(490.6)
    print d1.distance_source_origin__mm, d1.distance_origin_detector__mm, d1.distance_source_detector__mm

if 0:
    d = Data()
    d.set_name('detector_two_spheres_Astra_20150313')
    fn = d.filename
    d.load()
    d.set_distance_source_origin__mm(44)
    d.set_distance_origin_detector__mm(55)
    d.set_filename('/root/name/dataSetName.extension')
    
    print d.distance_source_detector__mm
    print d.shape
    print d.parent_path + d.name + d.data_format
    
    print d.angle_range__rad
    d.set_angle_range__rad(1)
    print d.angles__rad[-1]
