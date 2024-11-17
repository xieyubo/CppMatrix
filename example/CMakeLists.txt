add_executable(example
    main.cpp
    # ref.cpp
)
target_link_libraries(example PRIVATE
    gpu_matrix
)

add_subdirectory(mnist)