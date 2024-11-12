add_executable(example
    main.cpp
    # ref.cpp
)
target_link_libraries(example PRIVATE
    cnn
)

add_subdirectory(mnist)