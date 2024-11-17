add_library(gpu_matrix
    adapter.cpp
)
target_sources(gpu_matrix PUBLIC FILE_SET CXX_MODULES FILES
    adapter.cppm
    gpu_instance.cpp
    ref_ptr.cpp
    log.cpp
    matrix.cpp
    module.cpp
    promise.cpp
)
target_link_libraries(gpu_matrix INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/webgpu/libdawn.so
)