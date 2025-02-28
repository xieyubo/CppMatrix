module;

#include <cassert>
#include <format>
#include <functional>
#include <future>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <webgpu/webgpu.h>

import webgpu;

using namespace webgpu;

export module cpp_matrix:webgpu_matrix;
import :matrix_type;
import :std_patch;

namespace cpp_matrix::backend {

export template <MatrixElementType T>
class WebGpuMatrix {
    template <MatrixElementType R>
    friend WebGpuMatrix<R> ScalarOp(R v, const WebGpuMatrix<R>& m, char op);

public:
    using ElementType = T;

    WebGpuMatrix() = default;

    WebGpuMatrix(size_t row, size_t column)
        : m_row { row }
        , m_column { column }
    {
        m_paddingRow = (m_row + 3) & ~3;
        m_paddingColumn = (m_column + 3) & ~3;
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        m_pBuffer = adapter->CreateBuffer<T>(m_paddingRow, m_paddingColumn);

        // Zero out.
        std::vector<T> tmp(m_paddingRow * m_paddingColumn);
        wgpuQueueWriteBuffer(adapter->GetQueue(), m_pBuffer.get(), 0, tmp.data(), sizeof(T) * tmp.size());
    }

    size_t Row() const
    {
        return m_row;
    }

    size_t Column() const
    {
        return m_column;
    }

    WGPUBuffer GetBuffer() const
    {
        return m_pBuffer.get();
    }

    size_t BufferSize() const
    {
        return sizeof(T) * m_paddingRow * m_paddingColumn;
    }

    WebGpuMatrix& operator=(std::vector<T> data)
    {
        m_row = 1;
        m_column = data.size();
        m_paddingRow = 4;
        m_paddingColumn = (m_column + 3) & ~3;
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        m_pBuffer = adapter->CreateBuffer<T>(m_paddingRow, m_paddingColumn);
        Write(std::span<T> { data });
        return *this;
    }

    void Write(std::span<T> data)
    {
        std::vector<T> tmp(m_paddingRow * m_paddingColumn);
        for (auto row = 0; row < m_row; ++row) {
            for (auto column = 0; column < m_column; ++column) {
                auto i = IndexInMat4x4ArrayMemory(row, column);
                tmp.data()[i] = data.data()[row * m_column + column];
            }
        }
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        wgpuQueueWriteBuffer(adapter->GetQueue(), m_pBuffer.get(), 0, tmp.data(), sizeof(T) * tmp.size());
    }

    operator bool() const
    {
        return m_pBuffer;
    }

    WebGpuMatrix operator*(const WebGpuMatrix& other) const
    {
        if (m_column != other.m_row) {
            throw std::runtime_error { "Can't dot two matrixs" };
        }

        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = WebGpuMatrix { m_row, other.m_column };

        // Caculate mat4x4
        size_t N = (m_paddingColumn >> 2) * (other.m_paddingColumn >> 2) * (m_paddingRow >> 2);
        auto intermediaBuffer = adapter->CreateBuffer<T>(N * 4 * 4);
        if (N) {
            auto code = std::format(R"({0}
@group(0) @binding(0) var<storage, read_write> input1: array<mat4x4<{1}>>;
@group(0) @binding(1) var<storage, read_write> input2: array<mat4x4<{1}>>;
@group(0) @binding(2) var<storage, read_write> output: array<mat4x4<{1}>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {2}) {{
        var a_row_index = i / ({3} * {4});
        var a_col_index = i % {3};
        var a_index = a_row_index * {3} + a_col_index;

        var b_row_index = i % {3};
        var b_col_index = i / {3} % {4};
        var b_index = b_row_index * {4} + b_col_index;

        output[i] = transpose(transpose(input1[a_index]) * transpose(input2[b_index]));
    }}
}}
)",
                WgslFeatures(), WgslElementType(), N, (m_paddingColumn >> 2), (other.m_paddingColumn >> 2));
            auto parameters = std::vector<Parameter> {
                { GetBuffer(), BufferSize() },
                { other.GetBuffer(), other.BufferSize() },
                { intermediaBuffer.get(), sizeof(T) * N * 4 * 4 },
            };
            webgpu::Run(code, { parameters.begin(), parameters.end() }, N, 256);

            code = std::format(R"({0}
@group(0) @binding(0) var<storage, read_write> input: array<mat4x4<{1}>>;
@group(0) @binding(1) var<storage, read_write> output: array<mat4x4<{1}>>;
@compute @workgroup_size(1)
fn main() {{
    for (var i = 0; i < {2}; i = i + 1) {{
        output[i / {3}] = output[i / {3}] + input[i];
    }}
}}
)",
                WgslFeatures(), WgslElementType(), N, m_paddingColumn >> 2);
            parameters = std::vector<Parameter> {
                { intermediaBuffer.get(), sizeof(T) * N * 4 * 4 },
                { output.GetBuffer(), output.BufferSize() },
            };
            webgpu::Run(code, { parameters.begin(), parameters.end() }, 1, 1);
        }

        return output;
    }

    WebGpuMatrix operator+(const WebGpuMatrix& other) const
    {
        return ElementWiseAddOrSub(other, '+');
    }

    WebGpuMatrix& operator+=(const WebGpuMatrix& other)
    {
        *this = *this + other;
        return *this;
    }

    WebGpuMatrix operator+(T v) const
    {
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto vbuffer = adapter->CreateBuffer<T>(1);

        // Buffer need 4 bytes aligned, so always write 4 bytes even it is std::float16_t
        std::float32_t tmp { v };
        wgpuQueueWriteBuffer(adapter->GetQueue(), vbuffer.get(), 0, &v, sizeof(tmp));

        auto output = WebGpuMatrix { m_row, m_column };

        // Caculate mat4x4
        size_t N = (m_paddingRow >> 2) * m_paddingColumn;
        auto code = std::format(R"({0}
@group(0) @binding(0) var<storage, read_write> input1: array<vec4<{1}>>;
@group(0) @binding(1) var<storage, read_write> input2: {1};
@group(0) @binding(2) var<storage, read_write> output: array<vec4<{1}>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {2}) {{
        output[i] = input1[i] + input2;
    }}
}}
)",
            WgslFeatures(), WgslElementType(), N);
        auto parameters = std::vector<Parameter> {
            { GetBuffer(), BufferSize() },
            { vbuffer.get(), sizeof(std::float32_t) },
            { output.GetBuffer(), output.BufferSize() },
        };
        webgpu::Run(code, { parameters.begin(), parameters.end() }, N, 256);
        return output;
    }

    WebGpuMatrix operator-(const WebGpuMatrix& other) const
    {
        return ElementWiseAddOrSub(other, '-');
    }

    WebGpuMatrix Sigmoid() const
    {
        auto output = WebGpuMatrix { m_row, m_column };

        // Caculate mat4x4
        size_t N = (m_paddingRow >> 2) * m_paddingColumn;
        auto code = std::format(R"({0}
@group(0) @binding(0) var<storage, read_write> input: array<vec4<{1}>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<{1}>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {2}) {{
        output[i] = 1 / (1 + exp(-input[i]));
    }}
}}
)",
            WgslFeatures(), WgslElementType(), N);
        auto parameters = std::vector<Parameter> {
            { GetBuffer(), BufferSize() },
            { output.GetBuffer(), output.BufferSize() },
        };
        webgpu::Run(code, { parameters.begin(), parameters.end() }, N, 256);
        return output;
    }

    WebGpuMatrix Transpose() const
    {
        auto output = WebGpuMatrix { m_column, m_row };

        // Caculate mat4x4
        size_t N = (m_paddingRow >> 2) * (m_paddingColumn >> 2);
        auto code = std::format(R"({0}
@group(0) @binding(0) var<storage, read_write> input: array<mat4x4<{1}>>;
@group(0) @binding(1) var<storage, read_write> output: array<mat4x4<{1}>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {2}) {{
        output[(i % {4}) * {3} + (i / {4})] = transpose(input[i]);
    }}
}}
)",
            WgslFeatures(), WgslElementType(), N, m_paddingRow >> 2, m_paddingColumn >> 2);
        auto parameters = std::vector<Parameter> {
            { GetBuffer(), BufferSize() },
            { output.GetBuffer(), output.BufferSize() },
        };
        webgpu::Run(code, { parameters.begin(), parameters.end() }, N, 256);
        return output;
    }

    WebGpuMatrix ElementProduct(const WebGpuMatrix& other) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = WebGpuMatrix { m_row, m_column };

        // Caculate vec4x4
        size_t N = (m_paddingRow >> 2) * m_paddingColumn;
        if (N) {
            auto code = std::format(R"({0}
@group(0) @binding(0) var<storage, read_write> input1: array<vec4<{1}>>;
@group(0) @binding(1) var<storage, read_write> input2: array<vec4<{1}>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<{1}>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {2}) {{
        output[i] = input1[i] * input2[i];
    }}
}}
)",
                WgslFeatures(), WgslElementType(), N);
            auto parameters = std::vector<Parameter> {
                { GetBuffer(), BufferSize() },
                { other.GetBuffer(), other.BufferSize() },
                { output.GetBuffer(), output.BufferSize() },
            };
            webgpu::Run(code, { parameters.begin(), parameters.end() }, N, 256);
        }

        return output;
    }

    std::vector<T> Read() const
    {
        std::vector<T> out(m_row * m_column);
        MapBuffer([this, &out](const T* data) {
            for (auto y = 0u; y < m_row; ++y) {
                for (auto x = 0u; x < m_column; ++x) {
                    auto i = IndexInMat4x4ArrayMemory(y, x);
                    out.data()[y * m_column + x] = data[i];
                }
            }
        });
        return out;
    }

    T operator[](size_t row, size_t column) const
    {
        if (row >= m_row || column >= m_column) {
            throw std::runtime_error { "Out of range" };
        }

        T ret {};
        MapBuffer([this, row, column, &ret](const T* data) {
            auto i = IndexInMat4x4ArrayMemory(row, column);
            ret = data[i];
        });
        return ret;
    }

    WebGpuMatrix Relu() const
    {
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = WebGpuMatrix { m_row, m_column };

        // Caculate vec4x4
        size_t N = (m_paddingRow >> 2) * m_paddingColumn;
        if (N) {
            auto code = std::format(R"({0}
@group(0) @binding(0) var<storage, read_write> input1: array<vec4<{1}>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<{1}>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {2}) {{
        output[i] = max(input1[i], vec4<{1}>(0.0));
    }}
}}
)",
                WgslFeatures(), WgslElementType(), N);
            auto parameters = std::vector<Parameter> {
                { GetBuffer(), BufferSize() },
                { output.GetBuffer(), output.BufferSize() },
            };
            webgpu::Run(code, { parameters.begin(), parameters.end() }, N, 256);
        }

        return output;
    }

private:
    static constexpr const char* WgslElementType()
    {
        return std::is_same_v<T, std::float16_t> ? "f16" : "f32";
    }

    static constexpr const char* WgslFeatures()
    {
        return std::is_same_v<T, std::float16_t> ? "enable f16;" : "";
    }

    WebGpuMatrix ElementWiseAddOrSub(const WebGpuMatrix& other, char op) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        auto output = WebGpuMatrix { m_row, m_column };

        // Caculate mat4x4
        size_t N = (m_paddingRow >> 2) * (m_paddingColumn >> 2);
        if (N) {
            auto code = std::format(R"({0}
@group(0) @binding(0) var<storage, read_write> input1: array<mat4x4<{1}>>;
@group(0) @binding(1) var<storage, read_write> input2: array<mat4x4<{1}>>;
@group(0) @binding(2) var<storage, read_write> output: array<mat4x4<{1}>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {2}) {{
        output[i] = input1[i] {3} input2[i];
    }}
}}
)",
                WgslFeatures(), WgslElementType(), N, op);
            auto parameters = std::vector<Parameter> {
                { GetBuffer(), BufferSize() },
                { other.GetBuffer(), other.BufferSize() },
                { output.GetBuffer(), output.BufferSize() },
            };
            webgpu::Run(code, { parameters.begin(), parameters.end() }, N, 256);
        }

        return output;
    }

    int IndexInMat4x4ArrayMemory(int row, int column) const
    {
        return (((row >> 2) * (m_paddingColumn >> 2) + (column >> 2)) << 4) + (((row & 0x3) << 2) + (column & 0x3));
    }

    void MapBuffer(std::function<void(const T*)> callback) const
    {
        auto adapter = GpuInstance::GetInstance().GetAdapter();

        auto bufferSize = BufferSize();

        auto readbackBufferDescriptor = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
            .size = bufferSize,
        };

        auto pReadbackBuffer = gpu_ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> { wgpuDeviceCreateBuffer(
            adapter->GetDevice(), &readbackBufferDescriptor) };

        auto commandEncoder = wgpuDeviceCreateCommandEncoder(adapter->GetDevice(), nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, m_pBuffer.get(), 0, pReadbackBuffer.get(), 0, bufferSize);
        auto commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);

        auto submitPromise = std::promise<WGPUQueueWorkDoneStatus>();
        auto submitFuture = submitPromise.get_future();
        wgpuQueueSubmit(adapter->GetQueue(), 1, &commandBuffer);
        wgpuQueueOnSubmittedWorkDone(adapter->GetQueue(),
            { .mode = WGPUCallbackMode_AllowProcessEvents,
                .callback
                = [](WGPUQueueWorkDoneStatus status, void* userdata1,
                      void* userdata2) { ((std::promise<WGPUQueueWorkDoneStatus>*)userdata1)->set_value(status); },
                .userdata1 = &submitPromise });
        if (auto status = Wait(submitFuture); status != WGPUQueueWorkDoneStatus_Success) {
            throw std::runtime_error { "wgpuQueueOnSubmittedWorkDone failed." };
        }

        auto mapPromise = std::promise<WGPUMapAsyncStatus>();
        auto mapFuture = mapPromise.get_future();
        wgpuBufferMapAsync(pReadbackBuffer.get(), WGPUMapMode_Read, 0, bufferSize,
            { .mode = WGPUCallbackMode_AllowProcessEvents,
                .callback = [](WGPUMapAsyncStatus status, struct WGPUStringView message, void* userdata1,
                                void* userdata2) { ((std::promise<WGPUMapAsyncStatus>*)userdata1)->set_value(status); },
                .userdata1 = &mapPromise });
        if (auto status = Wait(mapFuture); status != WGPUMapAsyncStatus_Success) {
            throw std::runtime_error { "wgpuBufferMapAsync failed." };
        }

        const auto* pMappedData = (T*)wgpuBufferGetConstMappedRange(pReadbackBuffer.get(), 0, bufferSize);
        callback(pMappedData);
        wgpuBufferUnmap(pReadbackBuffer.get());
    }

    template <typename R>
    R Wait(std::future<R>& future) const
    {
        while (future.wait_for(std::chrono::milliseconds {}) != std::future_status::ready) {
            ProcessGpuInstanceEvents();
        }
        return future.get();
    }

    size_t m_row {};
    size_t m_column {};
    size_t m_paddingRow {};
    size_t m_paddingColumn {};
    gpu_ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> m_pBuffer {};
};

template <MatrixElementType T>
WebGpuMatrix<T> ScalarOp(T v, const WebGpuMatrix<T>& m, char op)
{
    auto adapter = GpuInstance::GetInstance().GetAdapter();
    auto vbuffer = adapter->CreateBuffer<T>(1);

    // Buffer need 4 bytes aligned, so always write 4 bytes even it is std::float16_t
    std::float32_t tmp { v };
    wgpuQueueWriteBuffer(adapter->GetQueue(), vbuffer.get(), 0, &v, sizeof(tmp));

    auto output = WebGpuMatrix<T> { m.m_row, m.m_column };

    // Caculate mat4x4
    size_t N = (m.m_paddingRow >> 2) * m.m_paddingColumn;
    auto code = std::format(R"({0}
@group(0) @binding(0) var<storage, read_write> input1: {1};
@group(0) @binding(1) var<storage, read_write> input2: array<vec4<{1}>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<{1}>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {2}) {{
        output[i] = input1 {3} input2[i];
    }}
}}
)",
        WebGpuMatrix<T>::WgslFeatures(), WebGpuMatrix<T>::WgslElementType(), N, op);
    auto parameters = std::vector<Parameter> {
        { vbuffer.get(), sizeof(std::float32_t) },
        { m.GetBuffer(), m.BufferSize() },
        { output.GetBuffer(), output.BufferSize() },
    };
    webgpu::Run(code, { parameters.begin(), parameters.end() }, N, 256);
    return output;
}

export template <MatrixElementType T>
WebGpuMatrix<T> operator-(T v, const WebGpuMatrix<T>& m)
{
    return ScalarOp(v, m, '-');
}

export template <MatrixElementType T>
WebGpuMatrix<T> operator*(T v, const WebGpuMatrix<T>& m)
{
    return ScalarOp(v, m, '*');
}
}