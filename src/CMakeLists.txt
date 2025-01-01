add_subdirectory(webgpu)

add_library(cpp_matrix)
target_sources(cpp_matrix PUBLIC FILE_SET CXX_MODULES FILES
    gpu_matrix.cpp
    host_matrix.cpp
    matrix.cpp
    module.cpp
)
target_link_libraries(cpp_matrix PUBLIC
    webgpu
)