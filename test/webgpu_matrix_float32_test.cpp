#include <gtest/gtest.h>

import cpp_matrix;

class WebGpuMatrixFloat32Test : public testing::Test {
public:
    WebGpuMatrixFloat32Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::WebGpuMatrix);
    }

    ~WebGpuMatrixFloat32Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(WebGpuMatrixFloat32Test, X)

using Matrix = cpp_matrix::Matrix<std::float32_t>;

#include "matrix_test.cpp"