#include <gtest/gtest.h>

import cpp_matrix;

#define MATRIX_TEST(X) TEST(WebGpuMatrixFloat32Test, X)

using Matrix = cpp_matrix::WebGpuMatrix<std::float32_t>;

#include "matrix_test.cpp"