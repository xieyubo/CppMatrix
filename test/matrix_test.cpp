#include <coroutine>
#include <gtest/gtest.h>

import cpp_matrix;

using namespace cpp_matrix;

TEST(MatrixTest, DefaultConstructor)
{
    Matrix x {};
    ASSERT_EQ(x.Row(), 0);
    ASSERT_EQ(x.Column(), 0);
}

TEST(MatrixTest, SetAbsoluteFloat)
{
    Matrix x {};
    x = 1.123f;
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 1);
}