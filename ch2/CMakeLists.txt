# 声明需要的cmake 最低版本
cmake_minimum_required(VERSION 2.8)
# 声明一个cmake工程
project(HelloSLAM)
# 添加一个可执行的程序
#add_executable(helloSLAM helloSLAM.cpp)
# add_library(hello helloSLAM.cpp)
add_library(hello_share libHelloSLAM.cpp)

add_executable(useLib useLib.cpp)
target_link_libraries(useLib hello_share)
