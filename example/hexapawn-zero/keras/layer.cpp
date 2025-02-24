module;

#include <functional>
#include <memory>

export module keras:layer;

namespace keras {

export class Layer {
public:
    Layer() = default;

    Layer(void* p, std::function<void(void*)> deleter)
        : m_obj { p, std::move(deleter) }
    {
    }

protected:
    std::shared_ptr<void> m_obj {};
};

export template <typename T>
class LayerImpl : public Layer {
public:
    LayerImpl(auto&&... args)
        : Layer { new T { std::forward<decltype(args)>(args)... }, [](void* p) { delete (T*)p; } }
    {
    }

protected:
    T& impl()
    {
        return *(T*)m_obj.get();
    }
};

}