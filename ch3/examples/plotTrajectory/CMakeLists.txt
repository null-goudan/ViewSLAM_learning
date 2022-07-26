cmake_minimum_required(VERSION 3.16)
project(plotTrajectory)
include_directories("/usr/include/eigen3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "release")

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})


