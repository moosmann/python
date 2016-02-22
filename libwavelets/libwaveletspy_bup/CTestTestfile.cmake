# CMake generated Testfile for 
# Source directory: /home/jmoosmann/git/LCR/libwavelets
# Build directory: /home/jmoosmann/git/LCR/libwavelets/libwaveletspy
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(libwaveletstest "/home/jmoosmann/git/LCR/libwavelets/libwaveletspy/bin/libwaveletstest")
SUBDIRS(libwavelets)
SUBDIRS(libwaveletstest)
SUBDIRS(libwaveletspy)
