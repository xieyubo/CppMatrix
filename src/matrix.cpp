module;

#include <span>
#include <variant>
#include <vector>

export module cpp_matrix:matrix;
import :gpu_matrix;
import :host_matrix;

namespace cpp_matrix {

export class Matrix {
public:
    Matrix()
        : m_matrix { GpuMatrix {} }
    {
    }

    Matrix(size_t row, size_t column)
        : m_matrix { GpuMatrix { row, column } }
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

    Matrix& operator=(float f)
    {
        if (auto p = std::get_if<HostMatrix>(&m_matrix)) {
            p->operator=(f);
        } else {
            std::get<GpuMatrix>(m_matrix).operator=(f);
        }
        return *this;
    }

private:
    std::variant<HostMatrix, GpuMatrix> m_matrix {};
};

}