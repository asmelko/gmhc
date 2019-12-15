cmake_minimum_required(VERSION 3.0.0)
project(gmhc CXX)

SET (CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -wd4514 -wd5045")

include_directories(include)

add_executable(gmhc main.cpp)

add_subdirectory(serial)