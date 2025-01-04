#include <gtest/gtest.h>

import cpp_matrix;

class CpuMatrixFloat32Test : public testing::Test {
public:
    CpuMatrixFloat32Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::CpuMatrix);
    }

    ~CpuMatrixFloat32Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(CpuMatrixFloat32Test, X)

using Matrix = cpp_matrix::Matrix<std::float32_t>;

#include "matrix_test.cpp"