import numpy as np
cimport cython
from numpy cimport ndarray, empty

cdef extern from "math.h":
    double cos(double)
    double sin(double)

def rotation(ndarray[double] theta):
    # I think the syntax for empty is the same in the cimported numpy.pxd,
    # should check
    cdef ndarray[double, ndim=2, mode="c"] R = empty( (3,3) )

cdef double cx = cos(theta[0]), cy = cos(theta[1]), cz = cos(theta[2])
    cdef double sx = sin(theta[0]), sy = sin(theta[1]), sz = sin(theta[2])

    with cython.boundscheck(False):
        R[0,0] = cx*cz - sx*cy*sz
        R[0,1] = cx*sz + sx*cy*cz
        R[0,2] = sx*sy

        R[1,0] = -sx*cz - cx*cy*sz
        R[1,1] = -sx*sz + cx*cy*cz
        R[1,2] = cx*sy

        R[2,0] = sy*sz
        R[2,1] = -sy*cz
        R[2,2] = cy

    return R