add_library(gpu
    adapter.cpp
)
target_sources(gpu PUBLIC FILE_SET CXX_MODULES FILES
    adapter.cppm
    buffer.cpp
    context.cpp
    dimension.cpp
    ref_ptr.cpp
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