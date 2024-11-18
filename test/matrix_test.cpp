#include <coroutine>
#include <gtest/gtest.h>

import cpp_matrix;

using namespace cpp_matrix;

class MatrixTest : public testing::Test {
public:
    MatrixTest()
    {
        Matrix::SetDefaultMatrixType(MatrixType::Auto);
    }
};

#ifndef MATRIX_TEST
#define MATRIX_TEST(X) TEST_F(MatrixTest, X)
#endif

MATRIX_TEST(DefaultConstructor)
{
    Matrix x {};
    ASSERT_EQ(x.Row(), 0);
    ASSERT_EQ(x.Column(), 0);
    ASSERT_FALSE(!!x);
}

MATRIX_TEST(SetAbsoluteFloat)
{
    Matrix x {};
    x = 1.123f;
    ASSERT_TRUE(!!x);
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 1);

    auto data = x.Read();
    ASSERT_EQ(data.size(), 1);
    ASSERT_FLOAT_EQ(data[0], 1.123f);
}