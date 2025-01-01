add_library(webgpu)
target_sources(webgpu PUBLIC FILE_SET CXX_MODULES FILES
    gpu_adapter.cpp
    gpu_instance.cpp
    gpu_ref_ptr.cpp
    module.cpp
)
target_link_libraries(webgpu INTERFACE
    ${WEBGPU_PATH}/libwebgpu_dawn.so
)