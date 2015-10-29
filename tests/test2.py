# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:01:55 2015

@author: jmoosmann
"""

# Temperature plotting script

# First import required modules
#import sys , os # Operating system modules
import numpy as np # Numerical python
import matplotlib.pyplot as plt # Matplotlib plotting

# Read in data from file
fname = "/home/jmoosmann/data/other/test.txt"
temperatures = np.loadtxt (fname,delimiter =" ", skiprows =1)

# Now plot temperature values
plt.plot ( temperatures [:,1], temperatures [:,0] ,"x",color ="green")
plt.show ()
print temperatures

# When finished , exit the script
#exit (0)
