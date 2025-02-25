#include <cmath>
#include <format>
#include <span>

static constexpr Matrix::ElementType operator""_mf(long double v)
{
    return v;
}

MATRIX_TEST(DefaultConstructor)
{
    Matrix x {};
    ASSERT_EQ(x.Row(), 0);
    ASSERT_EQ(x.Column(), 0);
}

MATRIX_TEST(CreateRandomMatrix)
{
    auto m = Matrix::Random(3, 4);
    ASSERT_EQ(m.Row(), 3);
    ASSERT_EQ(m.Column(), 4);

    auto data = m.Read();
    for (const auto& v : data) {
        ASSERT_TRUE(v >= 0 && v <= 1);
    }
}

MATRIX_TEST(SetAbsoluteValue)
{
    Matrix x {};
    x = 1.123_mf;
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 1);

    auto data = x.Read();
    ASSERT_EQ(data.size(), 1);
    ASSERT_FLOAT_EQ(data[0], 1.123_mf);
}

MATRIX_TEST(SetVector)
{
    std::vector<Matrix::ElementType> initData { 1.0_mf, 1.2_mf, 1.3_mf };

    Matrix x {};
    x = initData;
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 3);

    auto data = x.Read();
    ASSERT_EQ(data.size(), 3);
    ASSERT_FLOAT_EQ(data[0], 1.0_mf);
    ASSERT_FLOAT_EQ(data[1], 1.2_mf);
    ASSERT_FLOAT_EQ(data[2], 1.3_mf);
}

MATRIX_TEST(SetInitializeList)
{
    Matrix x {};
    x = std::initializer_list<Matrix::ElementType> { 1.0_mf, 1.2_mf, 1.3_mf };
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 3);

    auto data = x.Read();
    ASSERT_EQ(data.size(), 3);
    ASSERT_FLOAT_EQ(data[0], 1.0_mf);
    ASSERT_FLOAT_EQ(data[1], 1.2_mf);
    ASSERT_FLOAT_EQ(data[2], 1.3_mf);
}

MATRIX_TEST(SetSpan1)
{
    std::vector<Matrix::ElementType> initData { 1.0_mf, 1.2_mf, 1.3_mf };
    auto span = std::span<Matrix::ElementType, 3> { initData };

    Matrix x {};
    x = span;
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 3);

    auto data = x.Read();
    ASSERT_EQ(data.size(), 3);
    ASSERT_FLOAT_EQ(data[0], 1.0_mf);
    ASSERT_FLOAT_EQ(data[1], 1.2_mf);
    ASSERT_FLOAT_EQ(data[2], 1.3_mf);
}

MATRIX_TEST(SetSpan2)
{
    std::vector<Matrix::ElementType> initData { 1.0_mf, 1.2_mf, 1.3_mf };
    auto span = std::span<Matrix::ElementType> { initData };

    Matrix x {};
    x = span;
    ASSERT_EQ(x.Row(), 1);
    ASSERT_EQ(x.Column(), 3);

    auto data = x.Read();
    ASSERT_EQ(data.size(), 3);
    ASSERT_FLOAT_EQ(data[0], 1.0_mf);
    ASSERT_FLOAT_EQ(data[1], 1.2_mf);
    ASSERT_FLOAT_EQ(data[2], 1.3_mf);
}

MATRIX_TEST(ReadByRowAndColumn)
{
    Matrix x { 3, 2 };

    // clang-format off
    std::vector<Matrix::ElementType> initData {
        1.0f, 1.1f,
        2.0f, 2.2f,
        3.0f, 3.3f,
    };
    // clang-format on

    x.Write(std::span<Matrix::ElementType> { initData });
    ASSERT_EQ(x.Row(), 3);
    ASSERT_EQ(x.Column(), 2);

    ASSERT_FLOAT_EQ((x[0, 0]), 1.0_mf);
    ASSERT_FLOAT_EQ((x[0, 1]), 1.1_mf);
    ASSERT_FLOAT_EQ((x[1, 0]), 2.0_mf);
    ASSERT_FLOAT_EQ((x[1, 1]), 2.2_mf);
    ASSERT_FLOAT_EQ((x[2, 0]), 3.0_mf);
    ASSERT_FLOAT_EQ((x[2, 1]), 3.3_mf);
}

MATRIX_TEST(MatrixAdd)
{
    auto test = [](size_t row, size_t column, int base) {
        Matrix x { row, column };
        Matrix y { row, column };

        std::vector<Matrix::ElementType> initData(row * column);
        auto i = 1;
        for (auto& e : initData) {
            e = base + i++;
        }

        x.Write(std::span<Matrix::ElementType> { initData });
        y.Write(std::span<Matrix::ElementType> { initData });

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
            test(row, column, row * column);
        }
    }

    test(100, 100, 10);
    test(100, 100, 20);
    test(100, 100, 30);
    test(1000, 1000, 40);
    test(1000, 1000, 50);
    test(1000, 1000, 60);
    test(5000, 5000, 70);
    test(5000, 5000, 80);
    test(5000, 5000, 90);
}

MATRIX_TEST(MatrixSelfAdd)
{
    auto test = [](size_t row, size_t column) {
        Matrix x { row, column };
        Matrix y { row, column };

        std::vector<Matrix::ElementType> initData(row * column);
        auto i = 1;
        for (auto& e : initData) {
            e = i++;
        }

        x.Write(std::span<Matrix::ElementType> { initData });
        y.Write(std::span<Matrix::ElementType> { initData });

        x += y;
        ASSERT_EQ(x.Row(), row);
        ASSERT_EQ(x.Column(), column);

        auto res = x.Read();
        ASSERT_EQ(res.size(), row * column);
        for (auto r = 0; r < row; ++r) {
            for (auto c = 0; c < column; ++c) {
                ASSERT_FLOAT_EQ(res[r * column + c], 2 * initData[r * column + c]);
            }
        }
    };

    for (auto row = 1u; row <= 10; ++row) {
        for (auto column = 1u; column <= 10; ++column) {
            test(row, column);
        }
    }

    test(100, 100);
    test(1000, 1000);
    test(5000, 5000);
}

MATRIX_TEST(MatrixAddScalar)
{
    auto test = [](size_t row, size_t column) {
        Matrix x { row, column };

        std::vector<Matrix::ElementType> initData(row * column);
        auto i = 1;
        for (auto& e : initData) {
            e = i++;
        }

        x.Write(std::span<Matrix::ElementType> { initData });

        auto z = x + 0.5_mf;
        ASSERT_EQ(z.Row(), row);
        ASSERT_EQ(z.Column(), column);

        auto res = z.Read();
        ASSERT_EQ(res.size(), row * column);
        for (auto y = 0; y < row; ++y) {
            for (auto x = 0; x < column; ++x) {
                ASSERT_FLOAT_EQ(res[y * column + x], initData[y * column + x] + 0.5_mf);
            }
        }
    };

    for (auto row = 1u; row <= 10; ++row) {
        for (auto column = 1u; column <= 10; ++column) {
            test(row, column);
        }
    }

    test(100, 100);
    test(1000, 1000);
    test(5000, 5000);
}

MATRIX_TEST(MatrixSub)
{
    auto test = [](size_t row, size_t column, auto base) {
        Matrix x { row, column };
        Matrix y { row, column };

        std::vector<Matrix::ElementType> xInitData(row * column);
        std::vector<Matrix::ElementType> yInitData(row * column);
        for (auto i = 0; i < row * column; ++i) {
            xInitData[i] = (base + i) * 2;
            yInitData[i] = base + i;
        }

        x.Write(std::span<Matrix::ElementType> { xInitData });
        y.Write(std::span<Matrix::ElementType> { yInitData });

        auto z = x - y;
        ASSERT_EQ(z.Row(), row);
        ASSERT_EQ(z.Column(), column);

        auto res = z.Read();
        ASSERT_EQ(res.size(), row * column);
        for (auto i = 0; i < row * column; ++i) {
            ASSERT_FLOAT_EQ(res[i], base + i);
        }
    };

    for (auto row = 1u; row <= 10; ++row) {
        for (auto column = 1u; column <= 10; ++column) {
            test(row, column, row * column);
        }
    }

    test(100, 100, 0.1_mf);
    test(100, 100, 0.2_mf);
    test(100, 100, 0.3_mf);

    if (std::is_same_v<Matrix::ElementType, std::float16_t>) {
        // Ignore following test cases for float16_t, the precision is too low and
        // the cumulative error precision is big.
        return;
    }

    test(1000, 1000, 0.4_mf);
    test(1000, 1000, 0.5_mf);
    test(1000, 1000, 0.6_mf);
    test(5000, 5000, 0.7_mf);
    test(5000, 5000, 0.8_mf);
    test(5000, 5000, 0.9_mf);
}

MATRIX_TEST(MatrixSubScalar)
{
    auto test = [](size_t row, size_t column) {
        Matrix x { row, column };

        std::vector<Matrix::ElementType> initData(row * column);
        auto i = 1;
        for (auto& e : initData) {
            e = i++;
        }

        x.Write(std::span<Matrix::ElementType> { initData });

        auto z = x - 0.5_mf;
        ASSERT_EQ(z.Row(), row);
        ASSERT_EQ(z.Column(), column);

        auto res = z.Read();
        ASSERT_EQ(res.size(), row * column);
        for (auto y = 0; y < row; ++y) {
            for (auto x = 0; x < column; ++x) {
                ASSERT_FLOAT_EQ(res[y * column + x], initData[y * column + x] - 0.5_mf);
            }
        }
    };

    for (auto row = 1u; row <= 10; ++row) {
        for (auto column = 1u; column <= 10; ++column) {
            test(row, column);
        }
    }

    test(100, 100);
    test(1000, 1000);
    test(5000, 5000);
}

MATRIX_TEST(MatrixMul)
{
    if (std::is_same_v<Matrix::ElementType, std::float16_t>) {
        // Ignore this test for float16_t, the precision is too low and
        // the cumulative error precision is big.
        return;
    }

    auto createMatrix = [](auto N, auto M, auto base) {
        Matrix x { N, M };
        std::vector<Matrix::ElementType> initData(N * M);
        for (auto n = 0; n < N; ++n) {
            for (auto m = 0; m < M; ++m) {
                initData[n * M + m] = base + (n + 1) + ((m + 1) * 0.1_mf);
            }
        }
        x.Write(std::span<Matrix::ElementType> { initData });
        return x;
    };

    // NxM * MxP
    auto test = [&createMatrix](size_t n, size_t m, size_t p, Matrix::ElementType base) {
        auto x = createMatrix(n, m, base);
        auto y = createMatrix(m, p, base);

        auto z = x * y;
        ASSERT_EQ(z.Row(), n);
        ASSERT_EQ(z.Column(), p);

        auto res = z.Read();
        for (auto r = 0; r < n; ++r) {
            for (auto c = 0; c < p; ++c) {
                auto sum = 0._mf;
                for (auto i = 0; i < m; ++i) {
                    auto a = base + (r + 1) + ((i + 1) * 0.1_mf);
                    auto b = base + (i + 1) + ((c + 1) * 0.1_mf);
                    sum += a * b;
                }
                ASSERT_FLOAT_EQ((res[r * p + c]), sum);
            }
        }
    };

    for (auto n = 1u; n <= 16; ++n) {
        for (auto m = 1u; m <= 16; ++m) {
            for (auto p = 1u; p <= 16; ++p) {
                test(n, m, p, (n * 0.1f + m * 0.01f + n * 0.001f));
            }
        }
    }

    test(20, 20, 20, 0.1f);
    test(20, 20, 20, 0.2f);
    test(20, 20, 20, 0.3f);
    test(50, 50, 50, 0.1f);
    test(50, 50, 50, 0.2f);
    test(50, 50, 50, 0.3f);
}

MATRIX_TEST(MatrixElementProduct)
{
    if (std::is_same_v<Matrix::ElementType, std::float16_t>) {
        // Ignore this test for float16_t, the precision is too low and
        // the cumulative error precision is big.
        return;
    }

    auto test = [](size_t row, size_t column, auto base) {
        Matrix x { row, column };
        Matrix y { row, column };

        std::vector<Matrix::ElementType> xInitData(row * column);
        std::vector<Matrix::ElementType> yInitData(row * column);
        for (auto i = 0; i < row * column; ++i) {
            xInitData[i] = yInitData[i] = base + i;
        }

        x.Write(std::span<Matrix::ElementType> { xInitData });
        y.Write(std::span<Matrix::ElementType> { yInitData });

        auto z = x.ElementProduct(y);
        ASSERT_EQ(z.Row(), row);
        ASSERT_EQ(z.Column(), column);

        auto res = z.Read();
        ASSERT_EQ(res.size(), row * column);
        for (auto i = 0; i < row * column; ++i) {
            ASSERT_FLOAT_EQ(res[i], (base + i) * (base + i));
        }
    };

    for (auto row = 1u; row <= 10; ++row) {
        for (auto column = 1u; column <= 10; ++column) {
            test(row, column, 0.1_mf);
        }
    }

    test(100, 100, 0.1_mf);
    test(100, 100, 0.2_mf);
    test(100, 100, 0.3_mf);
}

MATRIX_TEST(MatrixScalarSub)
{
    auto test = [](size_t row, size_t column) {
        Matrix x { row, column };

        std::vector<Matrix::ElementType> initData(row * column);
        auto i = 1;
        for (auto& e : initData) {
            e = i++;
        }

        x.Write(std::span<Matrix::ElementType> { initData });

        auto z = 0.5_mf - x;
        ASSERT_EQ(z.Row(), row);
        ASSERT_EQ(z.Column(), column);

        auto res = z.Read();
        ASSERT_EQ(res.size(), row * column);
        for (auto y = 0; y < row; ++y) {
            for (auto x = 0; x < column; ++x) {
                ASSERT_FLOAT_EQ(res[y * column + x], 0.5_mf - initData[y * column + x]);
            }
        }
    };

    for (auto row = 1u; row <= 10; ++row) {
        for (auto column = 1u; column <= 10; ++column) {
            test(row, column);
        }
    }

    test(100, 100);
    test(1000, 1000);
    test(5000, 5000);
}

MATRIX_TEST(MatrixScalarMul)
{
    auto test = [](size_t row, size_t column) {
        Matrix x { row, column };

        std::vector<Matrix::ElementType> initData(row * column);
        auto i = 1;
        for (auto& e : initData) {
            e = i++;
        }

        x.Write(std::span<Matrix::ElementType> { initData });

        auto z = 0.1_mf * x;
        ASSERT_EQ(z.Row(), row);
        ASSERT_EQ(z.Column(), column);

        auto res = z.Read();
        ASSERT_EQ(res.size(), row * column);
        for (auto y = 0; y < row; ++y) {
            for (auto x = 0; x < column; ++x) {
                ASSERT_FLOAT_EQ(res[y * column + x], 0.1_mf * initData[y * column + x]);
            }
        }
    };

    for (auto row = 1u; row <= 10; ++row) {
        for (auto column = 1u; column <= 10; ++column) {
            test(row, column);
        }
    }

    test(100, 100);
    test(1000, 1000);
    test(5000, 5000);
}

MATRIX_TEST(MatrixSigmoid)
{
    Matrix x { 3, 1 };

    // clang-format off
    std::vector<Matrix::ElementType> initData {
        1.16_mf,
        0.42_mf,
        0.62_mf,
    };
    // clang-format on

    x.Write(std::span<Matrix::ElementType> { initData });

    auto z = x.Sigmoid();
    ASSERT_EQ(z.Row(), 3);
    ASSERT_EQ(z.Column(), 1);

    auto res = z.Read();
    ASSERT_FLOAT_EQ(res[0], 0.76133269_mf);
    ASSERT_FLOAT_EQ(res[1], 0.60348326_mf);
    ASSERT_FLOAT_EQ(res[2], 0.65021855_mf);
}

MATRIX_TEST(MatrixTranspose)
{
    auto test = [](size_t M, size_t N) {
        Matrix x { M, N };
        std::vector<Matrix::ElementType> initData(M * N);
        for (auto m = 0; m < M; ++m) {
            for (auto n = 0; n < N; ++n) {
                initData[m * N + n] = m * 100 + n;
            }
        }
        x.Write(std::span<Matrix::ElementType> { initData });

        auto z = x.Transpose();
        ASSERT_EQ(z.Row(), N);
        ASSERT_EQ(z.Column(), M);

        auto res = z.Read();
        for (auto n = 0; n < N; ++n) {
            for (auto m = 0; m < M; ++m) {
                ASSERT_FLOAT_EQ(res[n * M + m], m * 100 + n);
            }
        }
    };

    for (auto m = 1u; m <= 20; ++m) {
        for (auto n = 1u; n <= 20; ++n) {
            test(m, n);
        }
    }
}

MATRIX_TEST(Relu)
{
    auto test = [](size_t row, size_t column) {
        Matrix x { row, column };

        std::vector<Matrix::ElementType> xInitData(row * column);
        for (auto i = 0; i < row * column; ++i) {
            xInitData[i] = i % 2 ? (i * -0.1_mf) : (i * 0.1_mf);
        }

        x.Write(std::span<Matrix::ElementType> { xInitData });

        auto z = x.Relu();
        ASSERT_EQ(z.Row(), row);
        ASSERT_EQ(z.Column(), column);

        auto res = z.Read();
        ASSERT_EQ(res.size(), row * column);
        for (auto i = 0; i < row * column; ++i) {
            ASSERT_FLOAT_EQ(res[i], i % 2 ? 0 : (i * 0.1_mf));
        }
    };

    for (auto m = 1u; m <= 20; ++m) {
        for (auto n = 1u; n <= 20; ++n) {
            test(m, n);
        }
    }
}

#if ENABLE_CUDA
MATRIX_TEST(Pow)
{
    auto test = [](size_t row, size_t column, Matrix::ElementType e) {
        Matrix x { row, column };

        std::vector<Matrix::ElementType> xInitData(row * column);
        for (auto i = 0; i < row * column; ++i) {
            xInitData[i] = i % 2 ? (i * -0.1_mf) : (i * 0.1_mf);
        }

        x.Write(std::span<Matrix::ElementType> { xInitData });
        auto z = x.Pow(e);
        ASSERT_EQ(z.Row(), row);
        ASSERT_EQ(z.Column(), column);

        auto res = z.Read();
        ASSERT_EQ(res.size(), row * column);
        for (auto i = 0; i < row * column; ++i) {
            ASSERT_FLOAT_EQ(res[i], pow(xInitData[i], e));
        }
    };

    for (auto m = 1u; m <= 20; ++m) {
        for (auto n = 1u; n <= 20; ++n) {
            for (auto e = 1u; e <= 1; ++e) {
                test(m, n, e);
            }
        }
    }
}
#endif