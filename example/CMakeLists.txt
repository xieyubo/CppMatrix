add_executable(example
    main.cpp
    # ref.cpp
)
target_link_libraries(example PRIVATE
    cpp_matrix
)

add_subdirectory(mnist)