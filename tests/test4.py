# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:00:12 2015

@author: jmoosmann
"""

import astra
from jmutility import *

__metaclass__ = type


class Projector(object):
    
    def __init__(self, geometry_type='cone',
                 num_voxel=(100, 100, 100),
                 det_row_count=100, det_col_count=100,
                 angles=np.linspace(0, 2 * np.pi, 180, endpoint=False),
                 det_col_spacing=1.0, det_row_spacing=1.0,
                 source_origin=100.0, origin_detector=10.0,
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
        self.gpu_index = gpu_index
        
    def init(self, volume_data=1, projection_data=1):
        # Create volume geometry
        self.volume_geom = astra.create_vol_geom(self.num_voxel)
    
        # Create projection geometry
        self.projection_geom = astra.create_proj_geom(self.geometry_type,
                                           self.detector_spacing_x, self.detector_spacing_y,
                                           self.det_row_count, self.det_col_count,
                                           self.angles,
                                           self.source_origin, self.origin_detector)
    
        # Allocate and store volume data in ASTRA memory
        self.volume_id = astra.data3d.create('-vol', self.volume_geom, volume_data)
    
        # Allocate and store projection data in ASTRA memeory
        self.projection_id = astra.data3d.create('-sino', self.projection_geom, projection_data)
        
        # Create algorithm object: forward projector
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = self.projection_id
        cfg['VolumeDataId'] = self.volume_id
        self.forward_alg_id = astra.algorithm.create(cfg)
        
        # Create algorithm object: backward projector
        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['option'] = {'GPUindex': self.gpu_index}
        cfg['ProjectionDataId'] = self.projection_id
        cfg['ReconstructionDataId'] = self.volume_id
        self.backward_alg_id = astra.algorithm.create(cfg)
        
    def set_volume_data(self, volume_data):
        astra.data3d.store(self.volume_id, volume_data)
    
    def set_projection_data(self, projection_data):
        astra.data3d.store(self.projection_id, projection_data)
                
    def forward(self):
        astra.algorithm.run(self.forward_alg_id)
        self.projection_data = astra.data3d.get(self.projection_id)
        
    def backward(self):
        astra.algorithm.run(self.backward_alg_id)
        self.volume_data = astra.data3d.get(self.volume_id)
        
    def clear(self):
        astra.data3d.delete(self.volume_id)
        astra.data3d.delete(self.projection_id)
        astra.algorithm.delete(self.forward_alg_id)
        astra.algorithm.delete(self.backward_alg_id)

# SIRT  
astra.clear()
p = Projector()
p.init()
p.forward()
p.backward()


class Mmm:
    
    def __init__(self, num_voxel = (100, 100, 100)):
        self.num_voxel = num_voxel
        self.vol_geom = astra.create_vol_geom(self.num_voxel)
        self.volume_id = astra.data3d.create('-vol', self.vol_geom, 0)
    
    counter = 0
    num_voxel = (10, 10, 10)
    
    
    def setName(self, name):
        self.name = name
        Mmm.counter += 1
        
    def setData(self, data=(1,2,3)):
        self.data = data
        Mmm.counter += 1
   
class Forward(Mmm):
     # Create algorithm object
        cfg = astra.astra_dict('FB3D_CUDA')
        cfg['option'] = {'GPUindex': 0}
        #cfg['ReconstructionDataId'] = volume_id
        #alg_id = astra.algorithm.create(cfg)
    
#a = Mmm()
#a.setName('honigkuchepfer')
#a.setData()
#print a.data, a.counter
#
#a.counter = 0
#print a.counter, Mmm.counter
#
#b = Mmm()
#Mmm.setData(b, (3,6,4))
#print b.data, b.counter
#
#print Mmm.counter
#astra.clear()
#a = Mmm()
#b = Mmm()
#print "Volume ID: ", a.volume_id, b.volume_id
#print a.num_voxel, b.num_voxel
#a.volume_id = 0
#print "Volume ID: ", a.volume_id, b.volume_id
#Mmm.volume_id = 0
#print "Volume ID: ", a.volume_id, b.volume_id

ainfo()
