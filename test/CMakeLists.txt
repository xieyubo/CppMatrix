add_executable(gpu_test
    gpu_test.cpp
)
target_link_libraries(gpu_test
    cpp_matrix
    GTest::gtest_main
)
gtest_discover_tests(gpu_test)