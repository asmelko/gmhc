cmake_minimum_required(VERSION 3.0.0)
project(gmhc CXX)

add_executable(gmhc main.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

target_include_directories(gmhc PUBLIC include)