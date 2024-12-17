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

    // clang-format off
    std::vector<float> initData {
        1.0f, 1.1f,
        2.0f, 2.2f,
        3.0f, 3.3f,
    };
    // clang-format on

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

MATRIX_TEST(MatrixAdd)
{
    auto test = [](size_t row, size_t column) {
        Matrix x { row, column };
        Matrix y { row, column };

        std::vector<float> initData(row * column);
        auto i = 1;
        for (auto& e : initData) {
            e = i++;
        }

        x.Write(std::span<float> { initData });
        y.Write(std::span<float> { initData });

        auto z = x + y;
        ASSERT_EQ(z.Row(), row);
        ASSERT_EQ(z.Column(), column);

        auto res = z.Read();
        ASSERT_EQ(res.size(), row * column);
        for (auto y = 0; y < row; ++y) {
            for (auto x = 0; x < column; ++x) {
                ASSERT_FLOAT_EQ(res[y * column + x], 2 * initData[y * column + x]);
            }
        }
    };

    for (auto row = 1u; row <= 10; ++row) {
        for (auto column = 1u; column <= 10; ++column) {
            test(row, column);
        }
    }

    test(100, 100);

    // BUG!!
    // test(1000, 1000); <-- failed. Memory issue?
}

MATRIX_TEST(MatrixMul)
{
    auto createMatrix = [](auto N, auto M) {
        Matrix x { N, M };
        std::vector<float> initData(N * M);
        for (auto n = 0; n < N; ++n) {
            for (auto m = 0; m < M; ++m) {
                initData[n * M + m] = (n + 1) + ((m + 1) * 0.1);
            }
        }
        x.Write(std::span<float> { initData });
        return x;
    };

    // NxM * MxP
    for (auto n = 1u; n <= 4; ++n) {
        for (auto m = 1u; m <= 4; ++m) {
            auto x = createMatrix(n, m);

            for (auto p = 1u; p <= 4; ++p) {
                auto y = createMatrix(m, p);

                auto z = x * y;
                ASSERT_EQ(z.Row(), n);
                ASSERT_EQ(z.Column(), p);

                for (auto r = 0; r < n; ++r) {
                    for (auto c = 0; c < p; ++c) {
                        auto sum = 0.0f;
                        for (auto i = 0; i < m; ++i) {
                            sum += ((r + 1) + (i + 1) * 0.1) * ((i + 1) + (c + 1) * 0.1);
                        }
                        ASSERT_FLOAT_EQ((z[r, c]), sum);
                    }
                }
            }
        }
    }
}
