CMAKE_MINIMUM_REQUIRED(VERSION 3.8.0)
PROJECT(gmhc_package CXX CUDA)

include(GenerateExportHeader)
include_directories("${PROJECT_BINARY_DIR}")

ADD_LIBRARY(gmhclust_C SHARED "src/gmhclust_C.cpp")

generate_export_header(gmhclust_C)

set_property(TARGET gmhclust_C PROPERTY CUDA_ARCHITECTURES OFF)

TARGET_LINK_LIBRARIES(gmhclust_C para)

add_custom_target(gmhc_package
                  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/ ${PROJECT_BINARY_DIR}/package
                  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gmhclust_C> ${PROJECT_BINARY_DIR}/package/src)

add_dependencies(gmhc_package gmhclust_C para)

add_custom_command(TARGET gmhc_package POST_BUILD
    COMMAND R ARGS CMD build .
    COMMAND R ARGS CMD INSTALL gmhc_0.0.1.tar.gz
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/package
)

