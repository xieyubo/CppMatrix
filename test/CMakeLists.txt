add_executable(cpp_matrix_test
    cpu_matrix_float16_test.cpp
    cpu_matrix_float32_test.cpp
    gpu_matrix_float32_test.cpp
)

if(NOT ENABLE_CUDA)
    target_sources(cpp_matrix_test gpu_matrix_float16_test.cpp)
endif()

target_link_libraries(cpp_matrix_test
    cpp_matrix
    GTest::gtest_main
)
gtest_discover_tests(cpp_matrix_test)