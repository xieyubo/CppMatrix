module;

#include <utility>

export module keras:input;
import :layer;
import :shape;

namespace keras {

export class Input : public Layer {
    class Impl {
    public:
        Impl(Shape shape)
            : m_shape { std::move(shape) }
        {
        }

    private:
        Shape m_shape {};
    };

public:
    Input(Shape shape)
        : Layer { new Impl { std::move(shape) }, [](void* p) { delete (Impl*)p; } }
    {
    }
};

}