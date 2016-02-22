#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'jmoosmann'

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

# make a 4x4 array of random numbers:
a = numpy.random.randn(4, 4)

# most nVidia devices only support single precision
a = a.astype(numpy.float32)

# allocate memory on the device:
a_gpu = cuda.mem_alloc(a.nbytes)

# transfer the data to the GPU:
cuda.memcpy_htod(a_gpu, a)


# Executing a Kernel
mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)
# Find a reference to our pycuda.driver.Function and call it, specifying a_gpu as the argument, and a block size of 4x4
func = mod.get_function("doublify")
func(a_gpu, block=(4, 4, 1))

# fetch the data back from the GPU
a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print a_doubled
print a
