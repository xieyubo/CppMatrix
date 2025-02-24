module;

#include <functional>
#include <stdfloat>

import cpp_matrix;

export module keras:losses;
import :layer;

namespace keras::losses {

export using Loss = std::function<std::float16_t(Layer y_true, Layer y_pred)>;

export Loss CategoricalCrossentropy(bool from_logits)
{
    return [](Layer y_true, Layer y_pred) -> std::float16_t { return {}; };
}

}