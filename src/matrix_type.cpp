module;

#include <cstdint>
#include <type_traits>

export module cpp_matrix:matrix_type;
import :std_patch;

namespace cpp_matrix {

export template <typename T>
concept MatrixElementType = std::is_same_v<T, std::float32_t> || std::is_same_v<T, std::float16_t>;

}