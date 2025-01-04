module;

#include <stdfloat>

export module cpp_matrix:std_patch;

namespace std {

#if __STDCPP_FLOAT16_T__ != 1
export using float16_t = _Float16;
#endif

#if __STDCPP_FLOAT32_T__ != 1
export using float32_t = float;
#endif

static_assert(sizeof(std::float16_t) == 2);
static_assert(sizeof(std::float32_t) == 4);

}