#include <gtest/gtest.h>

import cpp_matrix;

#define MATRIX_TEST(X) TEST(WebGpuMatrixFloat16Test, X)

using Matrix = cpp_matrix::WebGpuMatrix<std::float16_t>;

#include "matrix_test.cpp"