cmake_minimum_required(VERSION 3.0)
project(undistortImage)

set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(OpenCV REQUIRED)
include_directories("/usr/include/eigen3")
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(undistortImage undistortimage.cpp)
target_link_libraries(undistortImage ${OpenCV_LIBS})
target_link_libraries(undistortImage ${Pangolin_LIBRARIES})