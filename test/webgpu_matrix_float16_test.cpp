#include <gtest/gtest.h>

import cpp_matrix;

class WebGpuMatrixFloat16Test : public testing::Test {
public:
    WebGpuMatrixFloat16Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::WebGpuMatrix);
    }

    ~WebGpuMatrixFloat16Test()
    {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::Auto);
    }
};

#define MATRIX_TEST(X) TEST_F(WebGpuMatrixFloat16Test, X)

using Matrix = cpp_matrix::Matrix<std::float16_t>;

#include "matrix_test.cpp"