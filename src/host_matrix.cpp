module;

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

    HostMatrix& operator=(float f)
    {
        m_row = 1;
        m_column = 1;
        m_data.resize(1);
        m_data[0] = f;
        return *this;
    }

    std::vector<float> Read() const
    {
        return m_data;
    }

    HostMatrix operator*(const HostMatrix& other) const
    {
        // TODO
        return {};
    }

private:
    size_t m_row {};
    size_t m_column {};
    std::vector<float> m_data;
};

}