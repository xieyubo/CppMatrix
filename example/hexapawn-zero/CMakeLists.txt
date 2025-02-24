add_subdirectory(keras)

add_library(hexapawn_zero)
target_sources(hexapawn_zero PUBLIC FILE_SET CXX_MODULES FILES
    common/game.cpp
    mnx_minimax.cpp
    hexapawn_zero.module.cpp
)

add_executable(mnx_generateTrainingData
    mnx_generateTrainingData.cpp
)
target_link_libraries(mnx_generateTrainingData PRIVATE
    hexapawn_zero
)

add_executable(sup_network
    sup_network.cpp
)
target_link_libraries(sup_network PRIVATE
    keras
)