file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
include_directories(../../../)	# faiss root directory

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)
    if(${name} MATCHES "GPU" AND (NOT BUILD_WITH_GPU))
	    message(STATUS "Skip GPU: ${source}")
    else()
        # target
        add_executable(${name} ${source})
        target_link_libraries(${name} ${faiss_lib})

        # Install
        install(TARGETS ${name} DESTINATION bin)
    endif()
endforeach(source)
