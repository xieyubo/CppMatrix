module;

#include <cstddef>
#include <initializer_list>
#include <utility>

export module keras:shape;

namespace keras {

export class Shape {
public:
    Shape(std::initializer_list<size_t> l)
        : m_l { std::move(l) }
    {
    }

private:
    std::initializer_list<size_t> m_l {};
};

}