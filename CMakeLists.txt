cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_ARCHITECTURES "native")
project(cpp_matrix CXX CUDA)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Download dependencies
set(WEBGPU_PATH ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/webgpu)

if (NOT EXISTS ${WEBGPU_PATH})
    file(MAKE_DIRECTORY ${WEBGPU_PATH})
endif()

if ((NOT EXISTS ${WEBGPU_PATH}/libwebgpu_dawn.so) OR (NOT EXISTS ${WEBGPU_PATH}/webgpu.h))
    MESSAGE("Download libwebgpu_dawn.so and webgpu.h")
    file(DOWNLOAD
        https://github.com/xieyubo/libwebgpu_dawn/releases/download/2025.01.31/libwebgpu_dawn.so
        ${WEBGPU_PATH}/libwebgpu_dawn.so
        SHOW_PROGRESS)
    file(DOWNLOAD
        https://github.com/xieyubo/libwebgpu_dawn/releases/download/2025.01.31/webgpu.h
        ${WEBGPU_PATH}/webgpu.h
        SHOW_PROGRESS)
endif()

add_compile_options(-g)
if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    add_compile_options(-DENABLE_CUDA=1)
endif()

include_directories(
    ${WEBGPU_PATH}/..
)

find_package(GTest REQUIRED)

enable_testing()

add_subdirectory(example)
add_subdirectory(src)
add_subdirectory(test)