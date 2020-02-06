CMAKE_MINIMUM_REQUIRED (VERSION 3.0.0)
include(GoogleTest)
project(serial CXX)

SET(CMAKE_CXX_STANDARD 17)

file(GLOB SERIAL_SRC "${PROJECT_SOURCE_DIR}/src/*.cpp")

ADD_LIBRARY(serial ${SERIAL_SRC})

target_include_directories(serial PUBLIC include)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

INCLUDE(external_blaspp)
TARGET_INCLUDE_DIRECTORIES(serial PUBLIC ${BLASPP_INCLUDE_DIR})
TARGET_LINK_DIRECTORIES(serial PUBLIC ${BLASPP_LIBS_DIR})

INCLUDE(external_lapackpp)
TARGET_INCLUDE_DIRECTORIES(serial PUBLIC ${LAPACKPP_INCLUDE_DIR})
TARGET_LINK_DIRECTORIES(serial PUBLIC ${LAPACKPP_LIBS_DIR})

ADD_DEPENDENCIES(lapackpp blaspp)
ADD_DEPENDENCIES(serial lapackpp)

TARGET_LINK_LIBRARIES(serial libblaspp.so lapack liblapackpp.a)

enable_testing()

ADD_EXECUTABLE(serial_test tests/test.cpp)

TARGET_LINK_LIBRARIES(serial_test serial gmock_main pthread)

gtest_discover_tests(serial_test)	