cmake_minimum_required(VERSION 2.8.2)

project(cub-download NONE)

include(ExternalProject)
ExternalProject_Add(cub
  GIT_REPOSITORY    https://github.com/NVIDIA/cub.git
  GIT_TAG           1.11.0
  CMAKE_ARGS        -DCUB_ENABLE_TESTING=OFF -DCUB_ENABLE_EXAMPLES=OFF -DCUB_ENABLE_HEADER_TESTING=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Get_Property(cub INSTALL_DIR)

set(CUB_INCLUDE_DIR ${INSTALL_DIR}/include)