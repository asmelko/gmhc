CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(blaspp_fetcher NONE)
INCLUDE(ExternalProject)

ExternalProject_Add(
  blaspp
  PREFIX             ${CMAKE_BINARY_DIR}/externals/blaspp
  HG_REPOSITORY      https://bitbucket.org/icl/blaspp
  HG_TAG             be9b9ea 
  TIMEOUT            10
  LOG_DOWNLOAD       ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  CMAKE_ARGS -DBLASPP_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Get_Property(blaspp INSTALL_DIR)

set(BLASPP_INCLUDE_DIR ${INSTALL_DIR}/include)

set(BLASPP_LIBS_DIR ${INSTALL_DIR}/lib)
