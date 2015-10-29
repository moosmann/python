# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import scipy.io as sio

class data:
    
    def __init__(self):
        self.slist = ['bernd']
        self.geometry = geometry()
        self.proj = np.array([])
        self.filename = os.getenv('HOME') + '/Dropbox/matlab/PhaseUnwrapping2D/IM.mat'
        self.fieldname = ''
    
    def append_slist(self, obj):
        self.slist.append(obj)
    
    def __load(self):
        assert os.path.isfile(self.filename), "File does not exist."
        self.proj = sio.loadmat(self.filename)
        return self.proj
        
    def load(self):
        """Load projections if not yet loaded. Does not reload data, if 
        proj already exists. Use reload to force loading of projections.
        
        rtype : numpy.array        
        """
        self.proj = self.proj or self.__load()
        return self.proj
    
    def reload_(self):
        self.proj = self.__load()
        
    def print_proj(self):
        print self.proj
    
class geometry:
    """ Geometric parameters of data acquisition in CT setup.
    
    """
    
    def __init__(self):
        self.name = ''
        self.origin_detector = 0
        self.angle_range = 0
        
    def set_name(self, name):
        self.name = name

a = data()
b = data()
g = geometry()
print g.name
g.set_name('eckhart')
print g.name
a.geometry.angle_range
print a.proj, a.proj.__class__
a.proj = 0
b = a.load()
print b, b.__class__
print a.filename
a.load()