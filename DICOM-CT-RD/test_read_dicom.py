# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:10:22 2015

@author: jmoosmann
"""

import os
import numpy
import pydicom
# from matplotlib import pyplot, cm

PathDicom = "/home/jmoosmann/data/mayo/axial-noFS/DICOM-CT-PD"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    print('dirName:{}\nsubdirList:{}\nlength fileList:{}'.format(
            dirName, subdirList, len(fileList)))
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
RefDs = pydicom.read_file(lstFilesDCM[0])

print('RefDs:\n type:{}\n Rows:{} \n Columns: {}'.format(
        type(RefDs), RefDs.Rows, RefDs.Columns))

tags = RefDs.keys()
print(tags[0])
print(RefDs[0x7029100b])
print(RefDs[0x70311003])
print(RefDs[0x7033100B])
print(RefDs[0x7033100C])
print(RefDs[0x7033100D])
print(RefDs[0x70371009])
print(RefDs[0x7037100A])
print(RefDs[0x70331013])
print(RefDs[0x70331061])


# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
# ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
