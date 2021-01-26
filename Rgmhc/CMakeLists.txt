CMAKE_MINIMUM_REQUIRED(VERSION 3.8.0)
PROJECT(Rgmhc CXX CUDA)

include(GenerateExportHeader)
include_directories("${PROJECT_BINARY_DIR}")


ADD_LIBRARY(gmhclust_C SHARED "src/gmhclust_C.cpp")
generate_export_header(gmhclust_C)

set_property(TARGET gmhclust_C PROPERTY CUDA_ARCHITECTURES OFF)

TARGET_LINK_LIBRARIES(gmhclust_C para)

#install(TARGETS foocpp DESTINATION lib)
