module;

#include "std_patch.h"
#include <cstdint>
#include <type_traits>

export module cpp_matrix:matrix_type;

namespace cpp_matrix {

export enum class MatrixType {
    Auto,
    CpuMatrix,
    GpuMatrix,
};

static MatrixType s_defaultMatrixType { MatrixType::Auto };

export void SetDefaultMatrixType(MatrixType type)
{
    s_defaultMatrixType = type;
}

export MatrixType GetDefaultMatrixType()
{
    return s_defaultMatrixType;
}

export template <typename T>
concept MatrixElementType = std::is_same_v<T, std::float32_t> || std::is_same_v<T, std::float16_t>;

}