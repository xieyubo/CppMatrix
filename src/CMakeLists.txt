add_library(cpp_matrix)
target_sources(cpp_matrix PUBLIC FILE_SET CXX_MODULES FILES
    gpu_adapter.cpp
    gpu_compute_core.cpp
    gpu_instance.cpp
    gpu_matrix.cpp
    gpu_ref_ptr.cpp
    host_matrix.cpp
    matrix.cpp
    module.cpp
)
target_link_libraries(cpp_matrix INTERFACE
    ${WEBGPU_PATH}/libwebgpu_dawn.so
)