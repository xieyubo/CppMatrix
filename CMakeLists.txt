cmake_minimum_required(VERSION 3.30)

project(webgpu CXX)

# Download dependencies
set(WEBGPU_PATH ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/webgpu)

if (NOT EXISTS ${WEBGPU_PATH})
    file(MAKE_DIRECTORY ${WEBGPU_PATH})
endif()

if ((NOT EXISTS ${WEBGPU_PATH}/libwebgpu_dawn.so) OR (NOT EXISTS ${WEBGPU_PATH}/webgpu.h))
    MESSAGE("Download libwebgpu_dawn.so and webgpu.h")
    file(DOWNLOAD
        https://github.com/xieyubo/libwebgpu_dawn/releases/download/2024.12.31/libwebgpu_dawn.so
        ${WEBGPU_PATH}/libwebgpu_dawn.so
        SHOW_PROGRESS)
    file(DOWNLOAD
        https://github.com/xieyubo/libwebgpu_dawn/releases/download/2024.12.31/webgpu.h
        ${WEBGPU_PATH}/webgpu.h
        SHOW_PROGRESS)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-g)

include_directories(
    ${WEBGPU_PATH}/..
)

find_package(GTest REQUIRED)

enable_testing()

add_subdirectory(example)
add_subdirectory(src)
add_subdirectory(test)