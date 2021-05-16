CMAKE_MINIMUM_REQUIRED(VERSION 3.8.0)
PROJECT(para CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES OFF)

execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/cmake/cuda_check.cu -o ${PROJECT_BINARY_DIR}/cuda_check)
execute_process(COMMAND ${PROJECT_BINARY_DIR}/cuda_check
                  RESULT_VARIABLE CUDA_RETURN_CODE
                  OUTPUT_VARIABLE ARCH)
if(${CUDA_RETURN_CODE} EQUAL 0)
	 message(STATUS "CUDA Architecture: ${ARCH}")
	 set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH}")
else()
    message(ERROR "unsupported architecture")
endif()

SET(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

FILE(GLOB_RECURSE PARA_SRC "${PROJECT_SOURCE_DIR}/src/*")

ADD_LIBRARY(para ${PARA_SRC})

TARGET_COMPILE_FEATURES(para PUBLIC cxx_std_14)

set_target_properties(para PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	POSITION_INDEPENDENT_CODE ON)

TARGET_INCLUDE_DIRECTORIES(para PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

TARGET_LINK_LIBRARIES(para cusolver cublas)

ADD_EXECUTABLE(gmhclust "main.cpp")

TARGET_LINK_LIBRARIES(gmhclust para)

IF (BUILD_GMHC_TESTS)
	INCLUDE(GoogleTest)

	FILE(GLOB_RECURSE TEST_FILES "${PROJECT_SOURCE_DIR}/tests/input/*")

	foreach(file_path ${TEST_FILES})
		get_filename_component(file ${file_path} NAME)
		configure_file(
					${PROJECT_SOURCE_DIR}/tests/input/${file}
					${PROJECT_BINARY_DIR}/${file}
					COPYONLY)
	endforeach()

	ADD_EXECUTABLE(para_test tests/test.cpp tests/serial_impl.cpp)

	TARGET_LINK_LIBRARIES(para_test para gmock_main)
	
	TARGET_INCLUDE_DIRECTORIES(para_test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

	GTEST_DISCOVER_TESTS(para_test)
ENDIF ()
