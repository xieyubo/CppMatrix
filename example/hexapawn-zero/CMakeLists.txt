add_library(hexapawn_zero)
target_sources(hexapawn_zero PUBLIC FILE_SET CXX_MODULES FILES
    common/game.cpp
    mnx_minimax.cpp
)

# add_executable(mnx_generateTrainingData
#    mnx_generateTrainingData.cpp
# )

#add_executable(hexapawn_zero
#    main.cpp
#)
#target_sources(hexapawn_zero PUBLIC FILE_SET CXX_MODULES FILES
#    mnx_generateTrainingData.py
#    # sup_network.cpp
#)
#target_link_libraries(hexapawn_zero PRIVATE
#    cpp_matrix
#)