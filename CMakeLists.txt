cmake_minimum_required(VERSION 3.30)

project(webgpu CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-g)

include_directories(
    thirdparty
)

add_subdirectory(gpu)
add_subdirectory(test)