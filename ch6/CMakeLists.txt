cmake_minimum_required(VERSION 3.0)
project(gaussNewton)

set(CMAKE_CXX_FLAGS "-std=c++14")
find_package(OpenCV REQUIRED)
include_directories("/usr/include/eigen3")

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})