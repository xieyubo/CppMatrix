module;

#include <coroutine>
#include <cstdlib>
#include <format>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>
#include <webgpu/webgpu.h>

export module cpp_matrix:matrix;
import :adapter;
import :promise;
import :ref_ptr;
import :gpu_matrix;
import :host_matrix;

namespace cpp_matrix {

export class Matrix {
public:
    Matrix()
        : m_matrix { GpuMatrix {} }
    {
    }

    Matrix(size_t row, size_t column)
        : m_matrix { GpuMatrix { row, column } }
        , m_row { row }
        , m_column { column }
    {
    }

    Matrix(size_t row, size_t column, Adapter adapter, WGPUBuffer buffer)
        : m_matrix { GpuMatrix { row, column } }
        , m_row { row }
        , m_column { column }
        , m_adapter { std::move(adapter) }
        , m_pBuffer { std::move(buffer) }
    {
    }

    template <size_t N>
    void Write(std::span<float, N> data)
    {
        return std::get<GpuMatrix>(m_matrix).Write(std::move(data));
    }

    Promise<std::vector<float>> Read()
    {
        return std::get<GpuMatrix>(m_matrix).Read();
    }

    size_t SizeInBytes() const
    {
        return sizeof(float) * m_row * m_column;
    }

    WGPUBuffer GetBuffer() const
    {
        return std::get<GpuMatrix>(m_matrix).GetBuffer();
    }

    operator bool() const
    {
        return std::get<GpuMatrix>(m_matrix).operator bool();
    }

    Promise<Matrix> operator*(const Matrix& other)
    {
        if (m_column != other.m_row) {
            throw std::runtime_error { "Can't dot two matrixs" };
        }

        // The max dimension supported by wgsl is 4x4, so if the matrix dimension is bigger than it,
        // we need split it.
        if (m_row <= 4 && m_column <= 4 && other.m_column <= 4) {
            // Perfect, no need split.
            return DotWithNoSplication(other);
        } else {
            throw std::runtime_error { "not supported" };
        }
    }

    size_t Row() const
    {
        if (auto p = std::get_if<HostMatrix>(&m_matrix)) {
            return p->Row();
        } else {
            return std::get<GpuMatrix>(m_matrix).Row();
        }
    }

    size_t Column() const
    {
        if (auto p = std::get_if<HostMatrix>(&m_matrix)) {
            return p->Column();
        } else {
            return std::get<GpuMatrix>(m_matrix).Column();
        }
    }

    Matrix& operator=(float f)
    {
        if (auto p = std::get_if<HostMatrix>(&m_matrix)) {
            p->operator=(f);
        } else {
            std::get<GpuMatrix>(m_matrix).operator=(f);
        }
        return *this;
    }

private:
    std::variant<HostMatrix, GpuMatrix> m_matrix {};

    Promise<Matrix> DotWithNoSplication(const Matrix& other)
    {
        auto output = m_adapter.CreateMatrix(m_row, other.m_column);
        auto k = std::vector<cpp_matrix::Matrix> { *this, other, output };
        auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: {};
@group(0) @binding(1) var<storage, read_write> input2: {};
@group(0) @binding(2) var<storage, read_write> output: {};
@compute @workgroup_size(1)
fn main() {{
    output = input1 * input2;
}}
)",
            GetWgslType(), other.GetWgslType(), output.GetWgslType());
        co_await m_adapter.Run(code.c_str(), { k.begin(), k.end() }, 1);
        co_return output;
    }

    std::string GetWgslType() const
    {
        if (m_row == 1 && m_column == 1) {
            return "f32";
        } else if (m_row == 1) {
            return std::format("array<f32, {}>", m_column);
        } else if (m_column == 1) {
            return std::format("vec{}f", m_row);
        } else {
            return std::format("mat{}x{}f", m_column, m_row);
        }
    }

    size_t m_row {};
    size_t m_column {};
    Adapter m_adapter {};
    ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> m_pBuffer {};
};

}