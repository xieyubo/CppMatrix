module;

#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

export module cpp_matrix:matrix;
import :cpu_matrix;
import :webgpu_matrix;
import :matrix_type;
import :std_patch;

namespace cpp_matrix {

template <typename T>
concept MatrixBackend = std::is_same_v<T, backend::CpuMatrix<std::float16_t>>
    || std::is_same_v<T, backend::CpuMatrix<std::float32_t>> || std::is_same_v<T, backend::WebGpuMatrix<std::float16_t>>
    || std::is_same_v<T, backend::WebGpuMatrix<std::float32_t>>;

template <MatrixBackend M>
class Matrix {
public:
    using ElementType = M::ElementType;

    friend Matrix operator-(ElementType v, const Matrix& m);
    friend Matrix operator*(ElementType v, const Matrix& m);

    /// @brief Create a matrix with random value (value will be between 0 and 1).
    static Matrix Random(size_t row, size_t column)
    {
        auto matrix = Matrix { row, column };
        std::vector<ElementType> initData(row * column);
        for (auto& v : initData) {
            v = std::min(std::rand() / (float)RAND_MAX, 1.f);
        }
        matrix.Write(std::span<ElementType> { initData });
        return matrix;
    }

    Matrix()
        : Matrix { 0, 0 }
    {
    }

    Matrix(size_t row, size_t column)
        : m_matrix { row, column }
    {
    }

    Matrix(size_t row, size_t column, std::span<ElementType> initData)
        : m_matrix { row, column }
    {
        Write(initData);
    }

    template <size_t N>
    void Write(std::span<ElementType, N> data)
    {
        m_matrix.Write(data);
    }

    std::vector<ElementType> Read() const
    {
        return m_matrix.Read();
    }

    Matrix operator+(const Matrix& other) const
    {
        return m_matrix + other.m_matrix;
    }

    Matrix& operator+=(const Matrix& other)
    {
        m_matrix += other.m_matrix;
        return *this;
    }

    Matrix operator-(const Matrix& other) const
    {
        return m_matrix - other.m_matrix;
    }

    Matrix operator+(ElementType v) const
    {
        return m_matrix + v;
    }

    Matrix operator-(ElementType v) const
    {
        return *this + (-v);
    }

    Matrix operator*(const Matrix& other) const
    {
        return m_matrix * other.m_matrix;
    }

    size_t Row() const
    {
        return m_matrix.Row();
    }

    size_t Column() const
    {
        return m_matrix.Column();
    }

    Matrix& operator=(std::vector<ElementType> data)
    {
        m_matrix = std::move(data);
        return *this;
    }

    Matrix& operator=(ElementType f)
    {
        return operator=(std::vector<ElementType> { f });
    }

    Matrix& operator=(std::span<ElementType> data)
    {
        return operator=(std::vector<ElementType> { data.begin(), data.end() });
    }

    Matrix Transpose() const
    {
        return m_matrix.Transpose();
    }

    Matrix Sigmoid() const
    {
        return m_matrix.Sigmoid();
    }

    Matrix ElementProduct(const Matrix& other) const
    {
        return m_matrix.ElementProduct(other.m_matrix);
    }

    Matrix Relu() const
    {
        return m_matrix.Relu();
    }

    float operator[](size_t row, size_t column) const
    {
        return m_matrix[row, column];
    }

private:
    Matrix(M m)
        : m_matrix { std::move(m) }
    {
    }

    M m_matrix {};
};

export template <MatrixElementType T>
using CpuMatrix = Matrix<backend::CpuMatrix<T>>;

export template <MatrixElementType T>
using WebGpuMatrix = Matrix<backend::WebGpuMatrix<T>>;

CpuMatrix<std::float16_t> operator-(std::float16_t v, const CpuMatrix<std::float16_t>& m)
{
    return operator-(v, m.m_matrix);
}

CpuMatrix<std::float32_t> operator-(std::float32_t v, const CpuMatrix<std::float32_t>& m)
{
    return operator-(v, m.m_matrix);
}

CpuMatrix<std::float16_t> operator*(std::float16_t v, const CpuMatrix<std::float16_t>& m)
{
    return operator*(v, m.m_matrix);
}

CpuMatrix<std::float32_t> operator*(std::float32_t v, const CpuMatrix<std::float32_t>& m)
{
    return operator*(v, m.m_matrix);
}

WebGpuMatrix<std::float16_t> operator-(std::float16_t v, const WebGpuMatrix<std::float16_t>& m)
{
    return operator-(v, m.m_matrix);
}

WebGpuMatrix<std::float32_t> operator-(std::float32_t v, const WebGpuMatrix<std::float32_t>& m)
{
    return operator-(v, m.m_matrix);
}

WebGpuMatrix<std::float16_t> operator*(std::float16_t v, const WebGpuMatrix<std::float16_t>& m)
{
    return operator*(v, m.m_matrix);
}

WebGpuMatrix<std::float32_t> operator*(std::float32_t v, const WebGpuMatrix<std::float32_t>& m)
{
    return operator*(v, m.m_matrix);
}

}