CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
PROJECT(gmhc CXX)

SET(CMAKE_CXX_STANDARD 17)

IF(MSVC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4 -wd4505")
    string(REGEX REPLACE " /W3" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ELSE ()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
ENDIF()

OPTION(BUILD_GMHC_TESTS "build tests" ON)
OPTION(BUILD_GMHC_SERIAL "build CPU serial implementation" ON)
OPTION(BUILD_GMHC_PARA "build GPU parallel implementation" ON)


IF (BUILD_GMHC_TESTS)
    # Download and unpack googletest at configure time
    configure_file(cmake/external_gtest.cmake googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
      message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
      message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                     ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                     EXCLUDE_FROM_ALL)
                 
    ENABLE_TESTING()
ENDIF (BUILD_GMHC_TESTS)


INCLUDE_DIRECTORIES(include)

IF (BUILD_GMHC_SERIAL)
    ADD_SUBDIRECTORY(serial)
ENDIF (BUILD_GMHC_SERIAL)


IF (BUILD_GMHC_PARA)
    ADD_SUBDIRECTORY(para)
    ADD_SUBDIRECTORY(para_timer)
ENDIF (BUILD_GMHC_PARA)

add_subdirectory(gmhc_package)