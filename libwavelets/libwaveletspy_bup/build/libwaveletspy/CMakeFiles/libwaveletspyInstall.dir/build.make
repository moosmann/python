# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jmoosmann/git/LCR/libwavelets

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build

# Utility rule file for libwaveletspyInstall.

# Include the progress variables for this target.
include libwaveletspy/CMakeFiles/libwaveletspyInstall.dir/progress.make

libwaveletspy/CMakeFiles/libwaveletspyInstall:
	cd /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/libwaveletspy && /usr/bin/cmake -E copy /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/__init__.py /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/bin/__init__.py

libwaveletspyInstall: libwaveletspy/CMakeFiles/libwaveletspyInstall
libwaveletspyInstall: libwaveletspy/CMakeFiles/libwaveletspyInstall.dir/build.make
	cd /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/libwaveletspy && /usr/bin/cmake -E copy /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/wavelets.py /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/bin/wavelets.py
	cd /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/libwaveletspy && /usr/bin/cmake -E copy /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/bin/liblibwaveletspy.so /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/bin/libwaveletspy.so
	cd /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/libwaveletspy && cd /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/bin && python /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/setup.py install
.PHONY : libwaveletspyInstall

# Rule to build all files generated by this target.
libwaveletspy/CMakeFiles/libwaveletspyInstall.dir/build: libwaveletspyInstall
.PHONY : libwaveletspy/CMakeFiles/libwaveletspyInstall.dir/build

libwaveletspy/CMakeFiles/libwaveletspyInstall.dir/clean:
	cd /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/libwaveletspy && $(CMAKE_COMMAND) -P CMakeFiles/libwaveletspyInstall.dir/cmake_clean.cmake
.PHONY : libwaveletspy/CMakeFiles/libwaveletspyInstall.dir/clean

libwaveletspy/CMakeFiles/libwaveletspyInstall.dir/depend:
	cd /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jmoosmann/git/LCR/libwavelets /home/jmoosmann/git/LCR/libwavelets/libwaveletspy /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/libwaveletspy /home/jmoosmann/git/LCR/libwavelets/libwaveletspy/build/libwaveletspy/CMakeFiles/libwaveletspyInstall.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libwaveletspy/CMakeFiles/libwaveletspyInstall.dir/depend

