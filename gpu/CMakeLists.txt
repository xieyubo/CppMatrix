add_library(gpu)
target_sources(gpu PUBLIC FILE_SET CXX_MODULES FILES
    adapter.cpp
    context.cpp
    gpu_ref_ptr.cpp
    instance.cpp
    log.cpp
    module.cpp
    promise.cpp
    shape.cpp
    tensor.cpp
)
target_link_libraries(gpu INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/webgpu/libdawn.so
)