module;

#include <cmath>
#include <stdexcept>
#include <vector>

export module cpp_matrix:cpu_matrix;

namespace cpp_matrix {

export class CpuMatrix {
    friend CpuMatrix operator-(float v, const CpuMatrix& m);
    friend CpuMatrix operator*(float v, const CpuMatrix& m);

public:
    CpuMatrix() = default;

    CpuMatrix(size_t row, size_t column)
        : m_row { row }
        , m_column { column }
        , m_data(row * column)
    {
    }

    size_t Row() const
    {
        return m_row;
    }

    size_t Column() const
    {
        return m_column;
    }

    CpuMatrix& operator=(std::vector<float> data)
    {
        m_row = 1;
        m_column = data.size();
        m_data = std::move(data);
        return *this;
    }

    void Write(std::vector<float> data)
    {
        if (m_row * m_column != data.size()) {
            throw std::runtime_error { "Elements size is not the same." };
        }

        m_data = std::move(data);
    }

    std::vector<float> Read() const
    {
        return m_data;
    }

    CpuMatrix operator+(const CpuMatrix& other) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        CpuMatrix res { m_row, m_column };
        auto* pR = res.m_data.data();
        auto* p1 = m_data.data();
        auto* p2 = other.m_data.data();
        for (auto i = 0u; i < m_row * m_column; ++i) {
            *pR++ = *p1++ + *p2++;
        }
        return res;
    }

    CpuMatrix& operator+=(const CpuMatrix& other)
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        auto* p1 = m_data.data();
        auto* p2 = other.m_data.data();
        for (auto i = 0u; i < m_row * m_column; ++i) {
            *p1++ += *p2++;
        }
        return *this;
    }

    CpuMatrix operator+(float v) const
    {
        CpuMatrix res { m_row, m_column };
        for (auto i = 0u; i < m_row * m_column; ++i) {
            res.m_data[i] = m_data[i] + v;
        }
        return res;
    }

    CpuMatrix operator-(const CpuMatrix& other) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        CpuMatrix res { m_row, m_column };
        auto* pR = res.m_data.data();
        auto* p1 = m_data.data();
        auto* p2 = other.m_data.data();
        for (auto i = 0u; i < m_row * m_column; ++i) {
            *pR++ = *p1++ - *p2++;
        }
        return res;
    }

    CpuMatrix operator*(const CpuMatrix& other) const
    {
        CpuMatrix res { m_row, other.m_column };
        for (auto y = 0; y < m_row; ++y) {
            for (auto x = 0; x < other.m_column; ++x) {
                auto sum = .0f;
                for (auto i = 0; i < m_column; ++i) {
                    sum += m_data[y * m_column + i] * other.m_data[i * other.m_column + x];
                }
                res.m_data[y * other.m_column + x] = sum;
            }
        }
        return res;
    }

    CpuMatrix Sigmoid() const
    {
        CpuMatrix res { m_row, m_column };
        for (auto i = 0u; i < m_row * m_column; ++i) {
            res.m_data[i] = 1.f / (1.f + std::exp(-m_data[i]));
        }
        return res;
    }

    CpuMatrix Transpose() const
    {
        CpuMatrix res { m_column, m_row };
        for (int c = 0u; c < m_column; ++c) {
            for (int r = 0u; r < m_row; ++r) {
                res.m_data[c * m_row + r] = m_data[r * m_column + c];
            }
        }
        return res;
    }

    CpuMatrix ElementProduct(const CpuMatrix& other)
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        CpuMatrix res { m_row, m_column };
        auto* pR = res.m_data.data();
        auto* p1 = m_data.data();
        auto* p2 = other.m_data.data();
        for (auto i = 0u; i < m_row * m_column; ++i) {
            *pR++ = *p1++ * *p2++;
        }
        return res;
    }

    float operator[](size_t row, size_t column) const
    {
        if (row >= m_row || column >= m_column) {
            throw std::runtime_error { "Out of range" };
        }

        return m_data[row * m_column + column];
    }

    size_t BufferSize() const
    {
        return sizeof(float) * m_row * m_column;
    }

private:
    size_t m_row {};
    size_t m_column {};
    std::vector<float> m_data;
};

export CpuMatrix operator-(float v, const CpuMatrix& m)
{
    CpuMatrix res { m.m_row, m.m_column };
    auto* pR = res.m_data.data();
    auto* p1 = m.m_data.data();
    for (auto i = 0u; i < m.m_row * m.m_column; ++i) {
        *pR++ = v - *p1++;
    }
    return res;
}

export CpuMatrix operator*(float v, const CpuMatrix& m)
{
    CpuMatrix res { m.m_row, m.m_column };
    auto* pR = res.m_data.data();
    auto* p1 = m.m_data.data();
    for (auto i = 0u; i < m.m_row * m.m_column; ++i) {
        *pR++ = v * *p1++;
    }
    return res;
}

}