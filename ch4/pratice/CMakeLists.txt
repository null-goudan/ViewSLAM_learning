cmake_minimum_required(VERSION 2.8.16)
project(Ch4_pratice)
find_package( Sophus REQUIRED )
# include_directories(${Sophus_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
INCLUDE_DIRECTORIES(${Sophus_INCLUDE_DIRS})

add_executable(draw_trajectory draw_trajectory.cpp)
target_link_libraries(draw_trajectory Sophus::Sophus)