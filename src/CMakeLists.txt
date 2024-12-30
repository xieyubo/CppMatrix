add_library(cpp_matrix)
target_sources(cpp_matrix PUBLIC FILE_SET CXX_MODULES FILES
    adapter.cpp
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
    ${WEBGPU_PATH}/libwebgpu_dawn.so
)