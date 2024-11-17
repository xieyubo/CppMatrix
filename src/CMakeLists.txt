add_library(cpp_matrix
    adapter.cpp
)
target_sources(cpp_matrix PUBLIC FILE_SET CXX_MODULES FILES
    adapter.cppm
    gpu_instance.cpp
    gpu_matrix.cpp
    host_matrix.cpp
    ref_ptr.cpp
    log.cpp
    matrix.cpp
    module.cpp
    promise.cpp
)
target_link_libraries(cpp_matrix INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/webgpu/libdawn.so
)