#include <gtest/gtest.h>

import cpp_matrix;

class GpuMatrixFloat16Test : public testing::Test {
public:
    GpuMatrixFloat16Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::GpuMatrix);
    }

    ~GpuMatrixFloat16Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(GpuMatrixFloat16Test, X)

using Matrix = cpp_matrix::Matrix<std::float16_t>;

#include "matrix_test.cpp"