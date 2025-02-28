#include <gtest/gtest.h>

import cpp_matrix;

#define MATRIX_TEST(X) TEST(CpuMatrixFloat32Test, X)

using Matrix = cpp_matrix::CpuMatrix<std::float32_t>;

#include "matrix_test.cpp"