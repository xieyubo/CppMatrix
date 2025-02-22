add_library(hexapawn_zero)
target_sources(hexapawn_zero PUBLIC FILE_SET CXX_MODULES FILES
    board.cpp
    module.cpp
)