module;

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

}