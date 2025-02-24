add_library(keras)
target_sources(keras PUBLIC FILE_SET CXX_MODULES FILES
    dense.cpp
    input.cpp
    keras.module.cpp
    layer.cpp
    losses.cpp
    model.cpp
    shape.cpp
)
target_link_libraries(keras PUBLIC
    cpp_matrix
)