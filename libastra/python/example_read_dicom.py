"""
Created on Wed Nov 11 18:10:22 2015

@author: jmoosmann
"""

import pydicom as dicom
import os
# import numpy
# from matplotlib import pyplot, cm


PathDicom = "/home/jmoosmann/data/mayo/DICOM-CT-PD"
PathDicomDict = "/home/jmoosmann/data/mayo/DICOM-CT-PD-dict_v8.txt"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))


fn = lstFilesDCM[0]
DicomDir = os.path.dirname(fn)
fn2 = DicomDir + \
      '/ACR_Axial_FFSoff_(Adult).601.RAW.20140626.152545.068988.1.05760.dcm'
print 'dcm file names:', fn2

# Get ref file
RefDs = dicom.read_file(fn2)

print 'RefDs class:', RefDs.__class__

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
print '(Rows, Columns, ', ConstPixelDims

# Load spacing values (in mm)
# ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]),
# float(RefDs.SliceThickness))

ds = RefDs
de = ds[0x20,0x11]
print 'DataElement',de
print 'DataElement.value', de.value
print ds.dir('Focal')

PathDicom = '/home/jmoosmann/data/CT/MANIX/MANIX/MANIX/CER-CT/'
fn = PathDicom + 'IM-0001-0001.dcm'
ds = dicom.read_file(fn2)
print ds.dir('Modality')
print ds.Modality