add_library(cnn
    adapter.cpp
)
target_sources(cnn PUBLIC FILE_SET CXX_MODULES FILES
    adapter.cppm
    buffer.cpp
    dimension.cpp
    ref_ptr.cpp
    instance.cpp
    log.cpp
    module.cpp
    promise.cpp
    shape.cpp
    tensor.cpp
)
target_link_libraries(cnn INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/webgpu/libdawn.so
)