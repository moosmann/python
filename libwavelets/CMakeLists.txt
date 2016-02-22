cmake_minimum_required (VERSION 2.6) 
project (libwavelets)

#Set binary dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# add FindMatlab module
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) 

set_property(GLOBAL PROPERTY USE_FOLDERS true)
enable_testing ()

# Enable or disable high precision
option(HIGH_PRECISION "Use double precision" TRUE)

if(HIGH_PRECISION)
	add_definitions(-DHIGH_PRECISION)
endif(HIGH_PRECISION)

# Enable or disable high precision
option(MEX_ENABLED "Build for matlab" FALSE)

if(MEX_ENABLED)
	find_package(Matlab REQUIRED)
endif()

# Python bindings
option(PYTHON_ENABLED "Create python bindings" FALSE)

if(PYTHON_ENABLED)
	set(Boost_USE_SHARED_LIBS   ON)
	set(Boost_USE_MULTITHREADED ON)
	find_package(Boost COMPONENTS
                 python
				 REQUIRED)
	include_directories(${Boost_INCLUDE_DIRS})
	link_directories(${Boost_LIBRARY_DIRS})
	
	# Find python
	find_package(PythonLibs REQUIRED)
	include_directories(${PYTHON_INCLUDE_DIRS})
	
	link_directories(${PYTHON_LIBRARIES})
endif(PYTHON_ENABLED)

# Option to enable or disable debugging and optimization
option(DEBUGGING "Enable debugging (in make)" FALSE)
option(OPTIMIZATION "Enable optimization (in make)" TRUE)

if(OPTIMIZATION)
	if(DEBUGGING)
		SET(CMAKE_BUILD_TYPE RelWithDebInfo)
	else(DEBUGGING)
		SET(CMAKE_BUILD_TYPE Release)
	endif(DEBUGGING)
else(OPTIMIZATION)
	if(DEBUGGING)
		SET(CMAKE_BUILD_TYPE Debug)
	else(DEBUGGING)
		SET(CMAKE_BUILD_TYPE None)
	endif(DEBUGGING)
endif(OPTIMIZATION)

# Add sub directories as needed
add_subdirectory (libwavelets)
add_subdirectory (libwaveletstest)
add_test (NAME libwaveletstest COMMAND libwaveletstest)
if(MEX_ENABLED)
	add_subdirectory (libwaveletsmex)
endif(MEX_ENABLED)
if(PYTHON_ENABLED)
	add_subdirectory (libwaveletspy)
endif(PYTHON_ENABLED)


