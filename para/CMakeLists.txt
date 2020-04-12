CMAKE_MINIMUM_REQUIRED(VERSION 3.8.0)
PROJECT(para CXX CUDA)

FILE(GLOB_RECURSE PARA_SRC "${PROJECT_SOURCE_DIR}/src/*")

ADD_LIBRARY(para ${PARA_SRC})

TARGET_COMPILE_FEATURES(para PUBLIC cxx_std_14)

SET(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_61,code=sm_61")

set_target_properties(para
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

TARGET_INCLUDE_DIRECTORIES(para PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

TARGET_LINK_LIBRARIES(para cublas)

ADD_EXECUTABLE(gmhclust "main.cpp")

TARGET_LINK_LIBRARIES(gmhclust para)

IF (BUILD_GMHC_TESTS)
	INCLUDE(GoogleTest)

	configure_file(
                ${PROJECT_SOURCE_DIR}/tests/input/big
                ${PROJECT_BINARY_DIR}/big
				COPYONLY)

	ADD_EXECUTABLE(para_test tests/test.cpp tests/serial_impl.cpp)

	TARGET_LINK_LIBRARIES(para_test para gmock_main)
	
	TARGET_INCLUDE_DIRECTORIES(para_test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

	GTEST_DISCOVER_TESTS(para_test)
ENDIF ()