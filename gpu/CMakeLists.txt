add_library(gpu)
target_sources(gpu PUBLIC FILE_SET CXX_MODULES FILES
    context.cpp
    module.cpp
)
target_link_libraries(gpu INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/webgpu/libdawn.so
)