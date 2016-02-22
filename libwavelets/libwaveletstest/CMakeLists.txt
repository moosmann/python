set(Headers
	test.h)

set(Sources
	test.c
	compimages3.c)

include_directories ("${PROJECT_SOURCE_DIR}/libwavelets")

add_executable (libwaveletstest ${Headers} ${Sources})

target_link_libraries(libwaveletstest libwavelets)

# Link against math lib
IF(UNIX)
  TARGET_LINK_LIBRARIES(libwaveletstest m)
ENDIF(UNIX)