CMAKE_MINIMUM_REQUIRED(VERSION 3.8.0)
PROJECT(gmhc_package CXX CUDA)

include(GenerateExportHeader)
include_directories("${PROJECT_BINARY_DIR}")

ADD_LIBRARY(c_gmhclust SHARED "src/c_gmhclust.cpp")

generate_export_header(c_gmhclust)

set_target_properties(c_gmhclust PROPERTIES
    CUDA_ARCHITECTURES OFF
    POSITION_INDEPENDENT_CODE ON)

TARGET_LINK_LIBRARIES(c_gmhclust para)

add_custom_target(gmhc_package
                  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/ ${PROJECT_BINARY_DIR}/package
                  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:c_gmhclust> ${PROJECT_BINARY_DIR}/package/src)

add_dependencies(gmhc_package c_gmhclust)

add_custom_command(TARGET gmhc_package POST_BUILD
    COMMAND R ARGS CMD build .
    COMMAND R ARGS CMD INSTALL gmhc_0.0.2.tar.gz
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/package
)
