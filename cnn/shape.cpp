module;

#include <vector>

export module gpu:shape;

namespace gpu {

export struct Shape {
    Shape() = default;

    Shape(const std::initializer_list<size_t> dimensions)
        : m_dimensions { dimensions }
    {
        size_t total { 1 };
        for (auto i : m_dimensions) {
            total *= i;
        }
        m_size = total;
    }

    size_t size() const
    {
        return m_size;
    }

private:
    std::vector<size_t> m_dimensions {};
    size_t m_size {};
};

}