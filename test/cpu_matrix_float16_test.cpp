#include <gtest/gtest.h>

import cpp_matrix;

#define MATRIX_TEST(X) TEST(CpuMatrixFloat16Test, X)

using Matrix = cpp_matrix::CpuMatrix<std::float16_t>;

#include "matrix_test.cpp"