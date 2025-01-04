#include <gtest/gtest.h>

import cpp_matrix;

class CpuMatrixFloat16Test : public testing::Test {
public:
    CpuMatrixFloat16Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::CpuMatrix);
    }

    ~CpuMatrixFloat16Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(CpuMatrixFloat16Test, X)

using Matrix = cpp_matrix::Matrix<std::float16_t>;

#include "matrix_test.cpp"