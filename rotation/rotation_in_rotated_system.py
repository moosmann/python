from __future__ import division
import numpy as np
from math import cos, sin, acos, atan2, sqrt

# rotation angle
theta = 0

# unit vector along rotation axis
axis = np.array([1, 1, 0])
axis = axis /sqrt(np.sum(axis**2))
a, b, c = axis

# step 1: translate space s.t. rotation axis passes through origin

# step 2: rotate space about the x axis so that the rotation axis lies in
# the xz plane

# projection onto yz
d = sqrt(b**2 + c**2)

if d ==0:
    rx = np.matrix(np.ones((3,3)))
    rxi = rx
else:
    rx = np.matrix([[c/d, -b/d, 0],
                [b/c, c/d, 0],
                [0, 0, 1]])
    rxi = np.matrix([[c / d, b / d, 0],
                 [-b / c, c / d, 0],
                 [0, 0, 1]])

# step 3: rotate space about the y axis so that the rotation axis lies along
#  the positive z axis
ry = np.matrix([[d, 0, -a],
                [0, 1, 0],
                [a, 0, d]])

ryi = np.matrix([[d, 0, a],
                 [0, 1, 0],
                 [-a, 0, d]])

# step 4: rotation about the z axis by t (theta)
rz = np.matrix([[cos(theta), -sin(theta), 0],
                [sin(theta), cos(theta), 0],
                [0, 0, 1]])


# f The complete transformation to rotate a point (x,y,z) about the rotation
#  axis to a new point (x`,y`,z`) is as follows, the forward transforms
# followed by the reverse transforms.

t = rxi * ryi * rz * ry * rx

v = np.matrix([1, 1, 0]).transpose()