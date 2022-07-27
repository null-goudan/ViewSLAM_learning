cmake_minimum_required(VERSION 3.0)
project(undistortImage)

set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(OpenCV REQUIRED)
add_executable(undistortImage undistortimage.cpp)
target_link_libraries(undistortImage ${OpenCV_LIBS})