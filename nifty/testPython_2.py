# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:11:34 2015

@author: jmoosmann
"""

from simplewrap import *
import numpy as np
import os, platform
import matplotlib.pyplot as plt

def show_slices(array3d):
    nx, ny, nz = array3d.shape
    fig = plt.figure('Sinogram')

    cm = plt.cm.Greys
    
    ax1 = fig.add_subplot(1, 3, 1)
    nn = np.round(nx/2) - 1
    im1 = ax1.imshow(array3d[nn, :, :], cmap=cm)
    ax1.set_title("yz-slice %u of %u" % (nn, nx))
    ax1.set_ylabel('ax1 ylabel')

    ax2 = fig.add_subplot(1, 3, 2)
    nn = np.round(ny/2) - 1
    ax2.imshow(array3d[:, nn, :], cmap=cm)
    ax2.set_title("xz-slice %u of %u" % (nn, ny))
    ax2.set_ylabel('ax2 ylabel')

    ax3 = fig.add_subplot(1, 3, 3)
    nn = np.round(nz/2) - 1
    ax3.imshow(array3d[:, :, nn], cmap=cm)
    ax3.set_title("xy-slice %u of %u" % (nn,nz))
    ax3.set_ylabel('ax3 ylable')
    
    #plt.annotate('annotation', xy=(2, 1), xytext=(2, 1))
    #plt.show()
    return fig, ax1, im1



__all__ = ['test_library_niftyrec_c',
'PET_project','PET_backproject','PET_project_compressed','PET_backproject_compressed',
'SPECT_project_parallelholes','SPECT_backproject_parallelholes',
'CT_project_conebeam','CT_backproject_conebeam','CT_project_parallelbeam','CT_backproject_parallelbeam',
'ET_spherical_phantom','ET_cylindrical_phantom','ET_spheres_ring_phantom', 
'TR_grid_from_box_and_affine', 'TR_resample_grid', 'TR_resample_box', 'TR_gradient_grid', 'TR_gradient_box', 'TR_transform_grid',
 'INTERPOLATION_LINEAR','INTERPOLATION_POINT'] 

library_name = "_et_array_interface"
niftyrec_lib_paths = [localpath(), filepath(__file__), './', '/usr/local/niftyrec/lib/', 'C:/Prorgam Files/NiftyRec/lib/'] 

INTERPOLATION_LINEAR = 0
INTERPOLATION_POINT  = 1


class ErrorInCFunction(Exception): 
    def __init__(self,msg,status,function_name): 
        self.msg = str(msg) 
        self.status = status
        self.function_name = function_name
        if self.status == status_io_error(): 
            self.status_msg = "IO Error"
        elif self.status == status_initialisation_error(): 
            self.status_msg = "Error with the initialisation of the C library"
        elif self.status == status_parameter_error(): 
            self.status_msg = "One or more of the specified parameters are not right"
        elif self.status == status_unhandled_error(): 
            self.status_msg = "Unhandled error, likely a bug. "
        else: 
            self.status_msg = "Unspecified Error"
    def __str__(self): 
        return "'%s' returned by the C Function '%s' (error code %d). %s"%(self.status_msg,self.function_name,self.status,self.msg)


def status_success(): 
    """Returns the value returned by the function calls to the library in case of success. """
    r = call_c_function( niftyrec_c.status_success, [{'name':'return_value',  'type':'uint', 'value':None}] ) 
    return r.return_value

def status_io_error(): 
    """Returns the integer value returned by the function calls to the library in case of IO error. """
    r = call_c_function( niftyrec_c.status_io_error, [{'name':'return_value',  'type':'uint', 'value':None}] ) 
    return r.return_value

def status_initialisation_error(): 
    """Returns the value returned by the function calls to the library in case of initialisation error. """
    r = call_c_function( niftyrec_c.status_initialisation_error, [{'name':'return_value',  'type':'uint', 'value':None}] ) 
    return r.return_value

def status_parameter_error(): 
    """Returns the value returned by the function calls to the library in case of parameter error. """
    r = call_c_function( niftyrec_c.status_parameter_error, [{'name':'return_value',  'type':'uint', 'value':None}] ) 
    return r.return_value

def status_unhandled_error(): 
    """Returns the value returned by the function calls to the library in case of unhandled error. """
    r = call_c_function( niftyrec_c.status_unhandled_error, [{'name':'return_value',  'type':'uint', 'value':None}] ) 
    return r.return_value


class LibraryNotFound(Exception): 
    def __init__(self,msg): 
        self.msg = msg 
    def __str__(self): 
        return "Library cannot be found: %s"%str(self.msg) 




if platform.system()=='Linux':
    sep = ":"
elif platform.system()=='Darwin':
    sep = ":"
elif platform.system()=='Windows':
    sep = ";"
if os.environ.has_key('LD_LIBRARY_PATH'): 
    niftyrec_lib_paths = niftyrec_lib_paths + os.environ['LD_LIBRARY_PATH'].split(sep)
if os.environ.has_key('DYLD_LIBRARY_PATH'): 
    niftyrec_lib_paths = niftyrec_lib_paths + os.environ['DYLD_LIBRARY_PATH'].split(sep)
if os.environ.has_key('PATH'): 
    niftyrec_lib_paths = niftyrec_lib_paths + os.environ['PATH'].split(sep)

(found,fullpath,path) = find_c_library(library_name,niftyrec_lib_paths) 
niftyrec_c = load_c_library(fullpath)

   
   
def SPECT_project_parallelholes(activity,cameras,attenuation=None,psf=None,background=0.0, background_attenuation=0.0, use_gpu=1, truncate_negative_values=0): 
    """SPECT projection; parallel-holes geometry. """
    #accept attenuation=None and psf=None: 
    if attenuation  is None: 
        attenuation = np.zeros((0,0,0)) 
    if psf is None: 
        psf = np.zeros((0,0,0)) 
    N_projections = cameras.shape[0]
    descriptor = [{'name':'activity',               'type':'array',   'value':activity }, 
                  {'name':'activity_size',          'type':'array',   'value':int32(activity.shape) }, 
                  {'name':'projection',             'type':'array',   'value':None,   'dtype':float32,  'size':(activity.shape[0],activity.shape[1],N_projections),  'order':"F"  }, 
                  {'name':'projection_size',        'type':'array',   'value':int32([N_projections, activity.shape[0], activity.shape[1]]) }, 
                  {'name':'cameras',                'type':'array',   'value':cameras,                  'order':"F" }, 
                  {'name':'cameras_size',           'type':'array',   'value':int32(cameras.shape) }, 
                  {'name':'psf',                    'type':'array',   'value':psf,                      'order':"F" }, 
                  {'name':'psf_size',               'type':'array',   'value':int32(psf.shape) }, 
                  {'name':'attenuation',            'type':'array',   'value':attenuation }, 
                  {'name':'attenuation_size',       'type':'array',   'value':int32(attenuation.shape) }, 
                  {'name':'background',             'type':'float',   'value':background }, 
                  {'name':'background_attenuation', 'type':'float',   'value':background_attenuation }, 
                  {'name':'use_gpu',                'type':'int',     'value':use_gpu }, 
                  {'name':'truncate_negative_values','type':'int',    'value':truncate_negative_values },  ]
    
    r = call_c_function( niftyrec_c.SPECT_project_parallelholes, descriptor ) 
    #if not r.status == status_success(): 
     #   raise ErrorInCFunction("The execution of 'SPECT_project_parallelholes' was unsuccessful.",r.status,'niftyrec_c.SPECT_project_parallelholes')
    return r.dictionary['projection'], r
       
       
N = 64
cameras = np.linspace(0,np.pi,120)
s = (N,N,N)
activity = np.ones(s)

#sino = NiftyRec.et_project(activity, cameras, psf=None, attenuation=None, gpu=1, background=0.0, background_attenuation=0.0, truncate_negative_values=1)
sino, r = SPECT_project_parallelholes(activity, cameras, use_gpu=0, background=1, background_attenuation=1.0,truncate_negative_values=1)   
show_slices(sino)
print sino.min(), sino.max()
       
       
       
       
       