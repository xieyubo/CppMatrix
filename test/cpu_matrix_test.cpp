#include <gtest/gtest.h>

import cpp_matrix;

class CpuMatrixTest : public testing::Test {
public:
    CpuMatrixTest()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::CpuMatrix);
    }

    ~CpuMatrixTest()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(CpuMatrixTest, X)

#include "matrix_test.cpp"