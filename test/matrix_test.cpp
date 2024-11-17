#include <coroutine>
#include <gtest/gtest.h>

import cpp_matrix;

using namespace cpp_matrix;

TEST(MatrixTest, DefaultConstructor)
{
    Matrix x {};
    ASSERT_EQ(x.Row(), 0);
    ASSERT_EQ(x.Column(), 0);
    ASSERT_FALSE(!!x);
}

TEST(MatrixTest, SetAbsoluteFloat)
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