#include <gtest/gtest.h>

import cpp_matrix;

using namespace cpp_matrix;

class GpuMatrixTest : public testing::Test {
public:
    GpuMatrixTest()
    {
        Matrix::SetDefaultMatrixType(MatrixType::GpuMatrix);
    }

    ~GpuMatrixTest()
    {
        Matrix::SetDefaultMatrixType(MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(GpuMatrixTest, X)

#include "matrix_test.cpp"