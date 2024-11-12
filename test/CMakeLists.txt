add_executable(test
    main.cpp
    # ref.cpp
)
target_link_libraries(test PRIVATE
    gpu
)