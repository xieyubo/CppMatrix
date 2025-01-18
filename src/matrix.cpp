module;

#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

export module cpp_matrix:matrix;
import :cpu_matrix;
import :gpu_matrix;
import :matrix_type;

namespace cpp_matrix {

export template <MatrixElementType T>
class Matrix {
    friend Matrix operator-(T v, const Matrix& m);
    friend Matrix operator*(T v, const Matrix& m);

public:
    using ElementType = T;

    /// @brief Create a matrix with random value (value will be between 0 and 1).
    static Matrix Random(size_t row, size_t column)
    {
        auto matrix = Matrix { row, column };
        std::vector<T> initData(row * column);
        for (auto& v : initData) {
            v = std::min(std::rand() / (float)RAND_MAX, 1.f);
        }
        matrix.Write(std::span<T> { initData });
        return matrix;
    }

    Matrix()
        : Matrix { 0, 0 }
    {
    }

    Matrix(size_t row, size_t column, MatrixType type = MatrixType::Auto)
        : m_matrix { CreateMatrix(type, row, column) }
    {
    }

    Matrix(size_t row, size_t column, std::span<T> initData, MatrixType type = MatrixType::Auto)
        : m_matrix { CreateMatrix(type, row, column) }
    {
        Write(initData);
    }

    template <size_t N>
    void Write(std::span<T, N> data)
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Write(std::vector<T> { data.begin(), data.end() });
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).Write(data);
        }
    }

    std::vector<T> Read() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Read();
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).Read();
        }
    }

    Matrix operator+(const Matrix& other) const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->operator+(std::get<CpuMatrix<T>>(other.m_matrix));
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).operator+(std::get<GpuMatrix<T>>(other.m_matrix));
        }
    }

    Matrix& operator+=(const Matrix& other)
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            p->operator+=(std::get<CpuMatrix<T>>(other.m_matrix));
        } else {
            std::get<GpuMatrix<T>>(m_matrix).operator+=(std::get<GpuMatrix<T>>(other.m_matrix));
        }
        return *this;
    }

    Matrix operator-(const Matrix& other) const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->operator-(std::get<CpuMatrix<T>>(other.m_matrix));
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).operator-(std::get<GpuMatrix<T>>(other.m_matrix));
        }
    }

    Matrix operator+(float v) const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->operator+(v);
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).operator+(v);
        }
    }

    Matrix operator-(float v) const
    {
        return *this + (-v);
    }

    Matrix operator*(const Matrix& other) const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->operator*(std::get<CpuMatrix<T>>(other.m_matrix));
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).operator*(std::get<GpuMatrix<T>>(other.m_matrix));
        }
    }

    size_t Row() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Row();
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).Row();
        }
    }

    size_t Column() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Column();
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).Column();
        }
    }

    Matrix& operator=(std::vector<T> data)
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            p->operator=(std::move(data));
        } else {
            std::get<GpuMatrix<T>>(m_matrix).operator=(std::move(data));
        }
        return *this;
    }

    Matrix& operator=(T f)
    {
        return operator=(std::vector<T> { f });
    }

    Matrix& operator=(std::span<T> data)
    {
        return operator=(std::vector<T> { data.begin(), data.end() });
    }

    Matrix Transpose() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Transpose();
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).Transpose();
        }
    }

    Matrix Sigmoid() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Sigmoid();
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).Sigmoid();
        }
    }

    Matrix ElementProduct(const Matrix& other) const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->ElementProduct(std::get<CpuMatrix<T>>(other.m_matrix));
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).ElementProduct(std::get<GpuMatrix<T>>(other.m_matrix));
        }
    }

    Matrix Relu() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Relu();
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).Relu();
        }
    }

    float operator[](size_t row, size_t column) const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->operator[](row, column);
        } else {
            return std::get<GpuMatrix<T>>(m_matrix).operator[](row, column);
        }
    }

private:
    template <typename... Args>
    static std::variant<CpuMatrix<T>, GpuMatrix<T>> CreateMatrix(MatrixType type, Args&&... args)
    {
        if (type == MatrixType::Auto) {
            type = GetDefaultMatrixType();
        }

        if (type == MatrixType::Auto) {
            // TODO: detect whether webgpu is avaliable or not.
            type = MatrixType::GpuMatrix;
        }

        if (type == MatrixType::GpuMatrix) {
            return GpuMatrix<T> { std::forward<Args>(args)... };
        } else {
            return CpuMatrix<T> { std::forward<Args>(args)... };
        }
    }

    Matrix(CpuMatrix<T> m)
        : m_matrix { std::move(m) }
    {
    }

    Matrix(GpuMatrix<T> m)
        : m_matrix { std::move(m) }
    {
    }

    std::variant<CpuMatrix<T>, GpuMatrix<T>> m_matrix {};
};

export Matrix<std::float32_t> operator-(std::float32_t v, const Matrix<std::float32_t>& m)
{
    if (auto p = std::get_if<CpuMatrix<std::float32_t>>(&m.m_matrix)) {
        return operator-(v, *p);
    } else {
        return operator-(v, std::get<GpuMatrix<std::float32_t>>(m.m_matrix));
    }
}

export Matrix<std::float32_t> operator*(std::float32_t v, const Matrix<std::float32_t>& m)
{
    if (auto p = std::get_if<CpuMatrix<std::float32_t>>(&m.m_matrix)) {
        return operator*(v, *p);
    } else {
        return operator*(v, std::get<GpuMatrix<std::float32_t>>(m.m_matrix));
    }
}

export Matrix<std::float16_t> operator-(std::float16_t v, const Matrix<std::float16_t>& m)
{
    if (auto p = std::get_if<CpuMatrix<std::float16_t>>(&m.m_matrix)) {
        return operator-(v, *p);
    } else {
        return operator-(v, std::get<GpuMatrix<std::float16_t>>(m.m_matrix));
    }
}

export Matrix<std::float16_t> operator*(std::float16_t v, const Matrix<std::float16_t>& m)
{
    if (auto p = std::get_if<CpuMatrix<std::float16_t>>(&m.m_matrix)) {
        return operator*(v, *p);
    } else {
        return operator*(v, std::get<GpuMatrix<std::float16_t>>(m.m_matrix));
    }
}

}