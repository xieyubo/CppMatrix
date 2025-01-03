#include <gtest/gtest.h>

import cpp_matrix;

using namespace cpp_matrix;

class CpuMatrixTest : public testing::Test {
public:
    CpuMatrixTest()
    {
        Matrix::SetDefaultMatrixType(MatrixType::CpuMatrix);
    }

    ~CpuMatrixTest()
    {
        Matrix::SetDefaultMatrixType(MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(CpuMatrixTest, X)

#include "matrix_test.cpp"