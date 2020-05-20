CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(lapackpp_fetcher NONE)
INCLUDE(ExternalProject)

ExternalProject_Add(
  lapackpp
  PREFIX             ${CMAKE_BINARY_DIR}/externals/lapackpp
  HG_REPOSITORY      https://bitbucket.org/icl/lapackpp
  HG_TAG             557bdae
  TIMEOUT            10
  LOG_DOWNLOAD       ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  CMAKE_ARGS -Dblaspp_DIR=${BLASPP_LIBS_DIR}/blaspp -DBUILD_LAPACKPP_TESTS=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Get_Property(lapackpp INSTALL_DIR)

set(LAPACKPP_INCLUDE_DIR ${INSTALL_DIR}/include)

set(LAPACKPP_LIBS_DIR ${INSTALL_DIR}/lib)
