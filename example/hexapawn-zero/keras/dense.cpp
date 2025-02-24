module;

#include <string>

export module keras:dense;
import :layer;

namespace keras {

class DenseImpl {
public:
    DenseImpl(size_t units, std::string name, std::string activation) {};

    void setInput(Layer input)
    {
        m_input = input;
    }

private:
    Layer m_input {};
};

export class Dense : public LayerImpl<DenseImpl> {
public:
    Dense(size_t units, std::string activation)
        : Dense(units, /*name=*/"", std::move(activation))
    {
    }

    Dense(size_t units, std::string name, std::string activation)
        : LayerImpl { units, std::move(name), std::move(activation) }
    {
    }

    Dense& operator()(Layer input)
    {
        impl().setInput(input);
        return *this;
    }
};

}