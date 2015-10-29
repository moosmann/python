# -*- coding: utf-8 -*-
#from NiftyRec import NiftyRec
from NiftyPy import NiftyRec
#import NiftyPy
import numpy as np
import math
#from scipy.linalg import norm
#import matplotlib.pyplot as plt

def mlem(activity_old,norm,sinogram,cameras,attenuation,psf,GPU,epsilon):
    projection = NiftyRec.et_project(activity_old,cameras,psf=None, attenuation=None, gpu=1,background=0.0,ackground_attenaution=0.0,truncate_negative_values=1)
    update = NiftyRec.et_backproject((sinogram+epsilon)/(projection+epsilon),cameras, psf=None, attenuation=None, gpu=1,background=0,background_attenuation=0.0,truncate_negative_values=1)
    activity_new = activity_old*(update+epsilon)
    activity_new = activity_new/(norm+epsilon)


N = 64
cameras = np.linspace(0,math.pi,120)
s = (N,N,N)
activity = np.ones(s)

#sino = NiftyRec.et_project(activity, cameras, psf=None, attenuation=None, gpu=1, background=0.0, background_attenuation=0.0, truncate_negative_values=1)
sino = NiftyRec.NiftyRec.SPECT_project_parallelholes(activity, cameras)