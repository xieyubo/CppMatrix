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

MATRIX_TEST(SetAbsoluteValue)
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

MATRIX_TEST(SetVector)
{
    std::vector<float> initData { 1.0f, 1.2f, 1.3f };

    Matrix x {};
    x = initData;
    ASSERT_TRUE(x);
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 3);

    auto data = x.Read();
    ASSERT_EQ(data.size(), 3);
    ASSERT_FLOAT_EQ(data[0], 1.0f);
    ASSERT_FLOAT_EQ(data[1], 1.2f);
    ASSERT_FLOAT_EQ(data[2], 1.3f);
}

MATRIX_TEST(SetInitializeList)
{
    Matrix x {};
    x = std::initializer_list<float> { 1.0f, 1.2f, 1.3f };
    ASSERT_TRUE(x);
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 3);

    auto data = x.Read();
    ASSERT_EQ(data.size(), 3);
    ASSERT_FLOAT_EQ(data[0], 1.0f);
    ASSERT_FLOAT_EQ(data[1], 1.2f);
    ASSERT_FLOAT_EQ(data[2], 1.3f);
}

MATRIX_TEST(SetSpan1)
{
    std::vector<float> initData { 1.0f, 1.2f, 1.3f };
    auto span = std::span<float, 3> { initData };

    Matrix x {};
    x = span;
    ASSERT_TRUE(x);
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 3);

    auto data = x.Read();
    ASSERT_EQ(data.size(), 3);
    ASSERT_FLOAT_EQ(data[0], 1.0f);
    ASSERT_FLOAT_EQ(data[1], 1.2f);
    ASSERT_FLOAT_EQ(data[2], 1.3f);
}

MATRIX_TEST(SetSpan2)
{
    std::vector<float> initData { 1.0f, 1.2f, 1.3f };
    auto span = std::span<float> { initData };

    Matrix x {};
    x = span;
    ASSERT_TRUE(x);
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 3);

    auto data = x.Read();
    ASSERT_EQ(data.size(), 3);
    ASSERT_FLOAT_EQ(data[0], 1.0f);
    ASSERT_FLOAT_EQ(data[1], 1.2f);
    ASSERT_FLOAT_EQ(data[2], 1.3f);
}

MATRIX_TEST(ReadByRowAndColumn)
{
    Matrix x { 3, 2 };

    std::vector<float> initData {
        1.0f, 1.1f,
        2.0f, 2.2f,
        3.0f, 3.3f,
    };

    x.Write(std::span<float> { initData });
    ASSERT_TRUE(x);
    ASSERT_EQ(x.Row(), 3);
    ASSERT_EQ(x.Column(), 2);

    ASSERT_FLOAT_EQ((x[0, 0]), 1.0f);
    ASSERT_FLOAT_EQ((x[0, 1]), 1.1f);
    ASSERT_FLOAT_EQ((x[1, 0]), 2.0f);
    ASSERT_FLOAT_EQ((x[1, 1]), 2.2f);
    ASSERT_FLOAT_EQ((x[2, 0]), 3.0f);
    ASSERT_FLOAT_EQ((x[2, 1]), 3.3f);
}