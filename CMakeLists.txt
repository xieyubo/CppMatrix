cmake_minimum_required(VERSION 3.30)

project(webgpu CXX)

# Download dependencies
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/webgpu/libdawn.so)
    file(DOWNLOAD
        https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn.so 
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/webgpu/libdawn.so
        SHOW_PROGRESS)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-g)

include_directories(
    thirdparty
)

find_package(GTest REQUIRED)

enable_testing()

add_subdirectory(example)
add_subdirectory(cnn)
add_subdirectory(test)