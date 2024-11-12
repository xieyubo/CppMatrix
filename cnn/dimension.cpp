module;

#include <vector>

export module gpu:dimension;

namespace gpu {

export class Dimension {
public:
    Dimension() = default;

    Dimension(const std::initializer_list<size_t> dimensions)
        : m_dimensions { dimensions }
    {
        size_t elements { 1 };
        for (auto i : m_dimensions) {
            elements *= i;
        }
        m_elements = elements;
    }

    size_t elements() const
    {
        return m_elements;
    }

private:
    std::vector<size_t> m_dimensions {};
    size_t m_elements {};
};

}