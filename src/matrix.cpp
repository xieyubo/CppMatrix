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
    friend Matrix operator-(float v, const Matrix& m);
    friend Matrix operator*(float v, const Matrix& m);

public:
    /// @brief Create a matrix with random value (value will be between 0 and 1).
    static Matrix Random(size_t row, size_t column)
    {
        auto matrix = Matrix { row, column };
        std::vector<float> initData(row * column);
        for (auto& v : initData) {
            v = std::min(std::rand() / (float)RAND_MAX, 1.f);
        }
        matrix.Write(std::span<float> { initData });
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

    Matrix(size_t row, size_t column, std::span<float> initData, MatrixType type = MatrixType::Auto)
        : m_matrix { CreateMatrix(type, row, column) }
    {
        Write(initData);
    }

    template <size_t N>
    void Write(std::span<float, N> data)
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Write(std::vector<float> { data.begin(), data.end() });
        } else {
            return std::get<GpuMatrix>(m_matrix).Write(data);
        }
    }

    std::vector<float> Read() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Read();
        } else {
            return std::get<GpuMatrix>(m_matrix).Read();
        }
    }

    Matrix operator+(const Matrix& other) const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->operator+(std::get<CpuMatrix<T>>(other.m_matrix));
        } else {
            return std::get<GpuMatrix>(m_matrix).operator+(std::get<GpuMatrix>(other.m_matrix));
        }
    }

    Matrix& operator+=(const Matrix& other)
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            p->operator+=(std::get<CpuMatrix<T>>(other.m_matrix));
        } else {
            std::get<GpuMatrix>(m_matrix).operator+=(std::get<GpuMatrix>(other.m_matrix));
        }
        return *this;
    }

    Matrix operator-(const Matrix& other) const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->operator-(std::get<CpuMatrix<T>>(other.m_matrix));
        } else {
            return std::get<GpuMatrix>(m_matrix).operator-(std::get<GpuMatrix>(other.m_matrix));
        }
    }

    Matrix operator+(float v) const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->operator+(v);
        } else {
            return std::get<GpuMatrix>(m_matrix).operator+(v);
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
            return std::get<GpuMatrix>(m_matrix).operator*(std::get<GpuMatrix>(other.m_matrix));
        }
    }

    size_t Row() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Row();
        } else {
            return std::get<GpuMatrix>(m_matrix).Row();
        }
    }

    size_t Column() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Column();
        } else {
            return std::get<GpuMatrix>(m_matrix).Column();
        }
    }

    Matrix& operator=(std::vector<float> data)
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            p->operator=(std::move(data));
        } else {
            std::get<GpuMatrix>(m_matrix).operator=(std::move(data));
        }
        return *this;
    }

    Matrix& operator=(float f)
    {
        return operator=(std::vector<float> { f });
    }

    template <size_t Extent = std::dynamic_extent>
    Matrix& operator=(std::span<float, Extent> data)
    {
        return operator=(std::vector<float> { data.begin(), data.end() });
    }

    Matrix Transpose() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Transpose();
        } else {
            return std::get<GpuMatrix>(m_matrix).Transpose();
        }
    }

    Matrix Sigmoid() const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->Sigmoid();
        } else {
            return std::get<GpuMatrix>(m_matrix).Sigmoid();
        }
    }

    Matrix ElementProduct(const Matrix& other)
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->ElementProduct(std::get<CpuMatrix<T>>(other.m_matrix));
        } else {
            return std::get<GpuMatrix>(m_matrix).ElementProduct(std::get<GpuMatrix>(other.m_matrix));
        }
    }

    float operator[](size_t row, size_t column) const
    {
        if (auto p = std::get_if<CpuMatrix<T>>(&m_matrix)) {
            return p->operator[](row, column);
        } else {
            return std::get<GpuMatrix>(m_matrix).operator[](row, column);
        }
    }

private:
    template <typename... Args>
    static std::variant<CpuMatrix<T>, GpuMatrix> CreateMatrix(MatrixType type, Args&&... args)
    {
        if (type == MatrixType::Auto) {
            type = GetDefaultMatrixType();
        }

        if (type == MatrixType::Auto) {
            // TODO: detect whether webgpu is avaliable or not.
            type = MatrixType::GpuMatrix;
        }

        if (type == MatrixType::GpuMatrix) {
            return GpuMatrix { std::forward<Args>(args)... };
        } else {
            return CpuMatrix<T> { std::forward<Args>(args)... };
        }
    }

    Matrix(CpuMatrix<T> m)
        : m_matrix { std::move(m) }
    {
    }

    Matrix(GpuMatrix m)
        : m_matrix { std::move(m) }
    {
    }

    std::variant<CpuMatrix<T>, GpuMatrix> m_matrix {};
};

export Matrix<std::float32_t> operator-(std::float32_t v, const Matrix<std::float32_t>& m)
{
    if (auto p = std::get_if<CpuMatrix<std::float32_t>>(&m.m_matrix)) {
        return operator-(v, *p);
    } else {
        return operator-(v, std::get<GpuMatrix>(m.m_matrix));
    }
}

export Matrix<std::float32_t> operator*(std::float32_t v, const Matrix<std::float32_t>& m)
{
    if (auto p = std::get_if<CpuMatrix<std::float32_t>>(&m.m_matrix)) {
        return operator*(v, *p);
    } else {
        return operator*(v, std::get<GpuMatrix>(m.m_matrix));
    }
}

}