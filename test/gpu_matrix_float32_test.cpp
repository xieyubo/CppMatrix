#include <gtest/gtest.h>

import cpp_matrix;

class GpuMatrixFloat32Test : public testing::Test {
public:
    GpuMatrixFloat32Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::GpuMatrix);
    }

    ~GpuMatrixFloat32Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(GpuMatrixFloat32Test, X)

using Matrix = cpp_matrix::Matrix<std::float32_t>;

#include "matrix_test.cpp"