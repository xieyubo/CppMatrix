#include <gtest/gtest.h>

import cpp_matrix;

using namespace cpp_matrix;

class HostMatrixTest : public testing::Test {
public:
    HostMatrixTest()
    {
        Matrix::SetDefaultMatrixType(MatrixType::HostMatrix);
    }

    ~HostMatrixTest()
    {
        Matrix::SetDefaultMatrixType(MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(HostMatrixTest, X)

#include "matrix_test.cpp"