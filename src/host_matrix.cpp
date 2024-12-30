module;

#include <stdexcept>
#include <vector>

export module cpp_matrix:host_matrix;

namespace cpp_matrix {

export class HostMatrix {
public:
    HostMatrix() = default;

    HostMatrix(size_t row, size_t column)
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

    HostMatrix& operator=(std::vector<float> data)
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

    HostMatrix operator+(const HostMatrix& other) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        HostMatrix res { m_row, m_column };
        auto* pR = res.m_data.data();
        auto* p1 = m_data.data();
        auto* p2 = other.m_data.data();
        for (auto i = 0u; i < m_row * m_column; ++i) {
            *pR++ = *p1++ + *p2++;
        }
        return res;
    }

    HostMatrix operator+(float v) const
    {
        HostMatrix res { m_row, m_column };
        for (auto i = 0u; i < m_row * m_column; ++i) {
            res.m_data[i] = m_data[i] + v;
        }
        return res;
    }

    HostMatrix operator*(const HostMatrix& other) const
    {
        HostMatrix res { m_row, other.m_column };
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

}