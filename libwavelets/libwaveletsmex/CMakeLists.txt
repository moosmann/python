set(Headers
	mexutils.h
	libwaveletsmex_export.h)

set(Sources
	mexutils.c)
	
include_directories ("${PROJECT_SOURCE_DIR}/libwavelets")
add_library(libwaveletsmex SHARED ${Headers} ${Sources})
target_link_libraries(libwaveletsmex libwavelets)

#Add mex files
set(MEX_FILES 
	wavelet_transform
	invwavelet_transform)

#Generate projects for each file
foreach(CPP_FILE ${MEX_FILES})

	# set up MATLAB libraries
	set(CPP_FILE_NAME ${CPP_FILE}.c)
	
	link_directories ("${PROJECT_BINARY_DIR}/libwavelets"
					  "${PROJECT_BINARY_DIR}/libwaveletsmex")
	include_directories(${MATLAB_INCLUDE_DIR} 
						"${PROJECT_SOURCE_DIR}/libwavelets"
						"${PROJECT_SOURCE_DIR}/libwaveletsmex")
	add_library(${CPP_FILE} SHARED ${CPP_FILE_NAME} ${CMAKE_SOURCE_DIR}/Matlabdef.def)
	target_link_libraries(${CPP_FILE} ${MATLAB_LIBRARIES} libwavelets)

	## Change naming for the correct MATLAB wrapper
	if(WIN32)
		if (CMAKE_CL_64)
			#Windows64
			SET_TARGET_PROPERTIES(${CPP_FILE} PROPERTIES SUFFIX .mexw64 LINK_FLAGS /export:mexFunction)
		else()
			#Windows32
			SET_TARGET_PROPERTIES(${CPP_FILE} PROPERTIES SUFFIX .mexw32 LINK_FLAGS /export:mexFunction)
		endif()
	else(WIN32)
		if(APPLE)
			if (CMAKE_SIZEOF_VOID_P MATCHES "8")
				#Linux64
				SET_TARGET_PROPERTIES(${CPP_FILE} PROPERTIES SUFFIX .mexmaci64)
			else()
				#Linux32
				SET_TARGET_PROPERTIES(${CPP_FILE} PROPERTIES SUFFIX .mexmaci32)
			endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
		else(APPLE)
			if (CMAKE_SIZEOF_VOID_P MATCHES "8")
				#Linux64
				SET_TARGET_PROPERTIES(${CPP_FILE} PROPERTIES SUFFIX .mexa64 PREFIX "")
			else()
				#Linux32
				SET_TARGET_PROPERTIES(${CPP_FILE} PROPERTIES SUFFIX .mexglx PREFIX "")
			endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
		endif(APPLE)
	endif(WIN32)

	SET_PROPERTY(TARGET ${CPP_FILE} PROPERTY FOLDER "mex")
endforeach()
