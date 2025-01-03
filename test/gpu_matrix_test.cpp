#include <gtest/gtest.h>

import cpp_matrix;

class GpuMatrixTest : public testing::Test {
public:
    GpuMatrixTest()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::GpuMatrix);
    }

    ~GpuMatrixTest()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(GpuMatrixTest, X)

#include "matrix_test.cpp"