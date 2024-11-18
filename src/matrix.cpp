module;

#include <span>
#include <variant>
#include <vector>

export module cpp_matrix:matrix;
import :gpu_matrix;
import :host_matrix;

namespace cpp_matrix {

export enum class MatrixType {
    Auto,
    GpuMatrix,
    HostMatrix,
};

export class Matrix {
public:
    static void SetDefaultMatrixType(MatrixType type)
    {
        s_defaultMatrixType = type;
    }

    Matrix()
        : Matrix { 0, 0 }
    {
    }

    Matrix(size_t row, size_t column, MatrixType type = MatrixType::Auto)
        : m_matrix { CreateMatrix(type, row, column) }
    {
    }

    template <size_t N>
    void Write(std::span<float, N> data)
    {
        return std::get<GpuMatrix>(m_matrix).Write(std::move(data));
    }

    std::vector<float> Read() const
    {
        if (auto p = std::get_if<HostMatrix>(&m_matrix)) {
            return p->Read();
        } else {
            return std::get<GpuMatrix>(m_matrix).Read();
        }
    }

    size_t SizeInBytes() const
    {
        return sizeof(float) * Row() * Column();
    }

    operator bool() const
    {
        return SizeInBytes() != 0;
    }

    Matrix operator*(const Matrix& other) const
    {
        if (auto p = std::get_if<HostMatrix>(&m_matrix)) {
            p->operator*(std::get<HostMatrix>(other.m_matrix));
        } else {
            std::get<GpuMatrix>(m_matrix).operator*(std::get<GpuMatrix>(other.m_matrix));
        }
        return *this;
    }

    size_t Row() const
    {
        if (auto p = std::get_if<HostMatrix>(&m_matrix)) {
            return p->Row();
        } else {
            return std::get<GpuMatrix>(m_matrix).Row();
        }
    }

    size_t Column() const
    {
        if (auto p = std::get_if<HostMatrix>(&m_matrix)) {
            return p->Column();
        } else {
            return std::get<GpuMatrix>(m_matrix).Column();
        }
    }

    Matrix& operator=(std::vector<float> data)
    {
        if (auto p = std::get_if<HostMatrix>(&m_matrix)) {
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

private:
    static MatrixType s_defaultMatrixType;

    template <typename... Args>
    static std::variant<HostMatrix, GpuMatrix> CreateMatrix(MatrixType type, Args&&... args)
    {
        if (type == MatrixType::Auto) {
            type = s_defaultMatrixType;
        }

        if (type == MatrixType::Auto) {
            // TODO: detect whether webgpu is avaliable or not.
            type = MatrixType::GpuMatrix;
        }

        if (type == MatrixType::GpuMatrix) {
            return GpuMatrix { std::forward<Args>(args)... };
        } else {
            return HostMatrix { std::forward<Args>(args)... };
        }
    }

    std::variant<HostMatrix, GpuMatrix> m_matrix {};
};

MatrixType Matrix::s_defaultMatrixType { MatrixType::Auto };

}