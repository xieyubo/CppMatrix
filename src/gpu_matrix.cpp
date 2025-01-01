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

export module cpp_matrix:gpu_matrix;

namespace cpp_matrix {

export class GpuMatrix {
    friend GpuMatrix operator-(float v, const GpuMatrix& m);
    friend GpuMatrix operator*(float v, const GpuMatrix& m);

public:
    static bool IsSupported(size_t row, size_t column)
    {
        const auto& limits = GpuInstance::GetInstance().GetAdapter()->GetLimits().limits;
        auto requiredBufferSize = row * column * sizeof(float);
        return requiredBufferSize <= limits.maxStorageBufferBindingSize && requiredBufferSize < limits.maxBufferSize;
    }

    GpuMatrix() = default;

    GpuMatrix(size_t row, size_t column)
        : m_row { row }
        , m_column { column }
    {
        if (!IsSupported(row, column)) {
            throw std::runtime_error { "dimension is not supported." };
        }

        m_paddingRow = (m_row + 3) & ~3;
        m_paddingColumn = (m_column + 3) & ~3;
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        m_pBuffer = adapter->CreateBuffer(m_paddingRow, m_paddingColumn);

        // Zero out.
        std::vector<float> tmp(m_paddingRow * m_paddingColumn);
        wgpuQueueWriteBuffer(adapter->GetQueue(), m_pBuffer.get(), 0, tmp.data(), sizeof(float) * tmp.size());
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
        return sizeof(float) * m_paddingRow * m_paddingColumn;
    }

    GpuMatrix& operator=(std::vector<float> data)
    {
        m_row = 1;
        m_column = data.size();
        m_paddingRow = 4;
        m_paddingColumn = (m_column + 3) & ~3;
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        m_pBuffer = adapter->CreateBuffer(m_paddingRow, m_paddingColumn);
        Write(std::span<float> { data });
        return *this;
    }

    void Write(std::span<float> data)
    {
        std::vector<float> tmp(m_paddingRow * m_paddingColumn);
        for (auto row = 0; row < m_row; ++row) {
            for (auto column = 0; column < m_column; ++column) {
                auto i = IndexInMat4x4ArrayMemory(row, column);
                tmp.data()[i] = data.data()[row * m_column + column];
            }
        }
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        wgpuQueueWriteBuffer(adapter->GetQueue(), m_pBuffer.get(), 0, tmp.data(), sizeof(float) * tmp.size());
    }

    operator bool() const
    {
        return m_pBuffer;
    }

    GpuMatrix operator*(const GpuMatrix& other) const
    {
        if (m_column != other.m_row) {
            throw std::runtime_error { "Can't dot two matrixs" };
        }

        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = GpuMatrix { m_row, other.m_column };

        // Caculate mat4x4
        size_t N = (m_paddingColumn >> 2) * (other.m_paddingColumn >> 2) * (m_paddingRow >> 2);
        auto intermediaBuffer = adapter->CreateBuffer(N * 4 * 4);
        if (N) {
            auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: array<mat4x4f>;
@group(0) @binding(1) var<storage, read_write> input2: array<mat4x4f>;
@group(0) @binding(2) var<storage, read_write> output: array<mat4x4f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {0}) {{
        var a_row_index = i / ({1} * {2});
        var a_col_index = i % {1};
        var a_index = a_row_index * {1} + a_col_index;

        var b_row_index = i % {1};
        var b_col_index = i / {1} % {2};
        var b_index = b_row_index * {2} + b_col_index;

        output[i] = transpose(transpose(input1[a_index]) * transpose(input2[b_index]));
    }}
}}
)",
                N, (m_paddingColumn >> 2), (other.m_paddingColumn >> 2));
            auto parameters = std::vector<Parameter> {
                { GetBuffer(), BufferSize() },
                { other.GetBuffer(), other.BufferSize() },
                { intermediaBuffer.get(), sizeof(float) * N * 4 * 4 },
            };
            adapter->Execute(code, { parameters.begin(), parameters.end() }, N, 256);

            code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input: array<mat4x4f>;
@group(0) @binding(1) var<storage, read_write> output: array<mat4x4f>;
@compute @workgroup_size(1)
fn main() {{
    for (var i = 0; i < {0}; i = i + 1) {{
        output[i / {1}] = output[i / {1}] + input[i];
    }}
}}
)",
                N, m_paddingColumn >> 2);
            parameters = std::vector<Parameter> {
                { intermediaBuffer.get(), sizeof(float) * N * 4 * 4 },
                { output.GetBuffer(), output.BufferSize() },
            };
            adapter->Execute(code, { parameters.begin(), parameters.end() }, 1, 1);
        }

        return output;
    }

    GpuMatrix
    operator+(const GpuMatrix& other) const
    {

        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = GpuMatrix { m_row, m_column };

        // Caculate mat4x4
        size_t N = (m_paddingRow >> 2) * (m_paddingColumn >> 2);
        if (N) {
            auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: array<mat4x4f>;
@group(0) @binding(1) var<storage, read_write> input2: array<mat4x4f>;
@group(0) @binding(2) var<storage, read_write> output: array<mat4x4f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {}) {{
        output[i] = input1[i] + input2[i];
    }}
}}
)",
                N);
            auto parameters = std::vector<Parameter> {
                { GetBuffer(), BufferSize() },
                { other.GetBuffer(), other.BufferSize() },
                { output.GetBuffer(), output.BufferSize() },
            };
            adapter->Execute(code, { parameters.begin(), parameters.end() }, N, 256);
        }

        return output;
    }

    GpuMatrix& operator+=(const GpuMatrix& other)
    {
        *this = *this + other;
        return *this;
    }

    GpuMatrix operator+(float v) const
    {
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto vbuffer = adapter->CreateBuffer(1);
        wgpuQueueWriteBuffer(adapter->GetQueue(), vbuffer.get(), 0, &v, sizeof(float));

        auto output = GpuMatrix { m_row, m_column };

        // Caculate mat4x4
        size_t N = (m_paddingRow >> 2) * m_paddingColumn;
        auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> input2: f32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {}) {{
        output[i] = input1[i] + input2;
    }}
}}
)",
            N);
        auto parameters = std::vector<Parameter> {
            { GetBuffer(), BufferSize() },
            { vbuffer.get(), sizeof(float) },
            { output.GetBuffer(), output.BufferSize() },
        };
        adapter->Execute(code, { parameters.begin(), parameters.end() }, N, 256);
        return output;
    }

    GpuMatrix operator-(const GpuMatrix& other) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = GpuMatrix { m_row, m_column };

        // Caculate mat4x4
        size_t N = (m_paddingRow >> 2) * (m_paddingColumn >> 2);
        if (N) {
            auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: array<mat4x4f>;
@group(0) @binding(1) var<storage, read_write> input2: array<mat4x4f>;
@group(0) @binding(2) var<storage, read_write> output: array<mat4x4f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {}) {{
        output[i] = input1[i] - input2[i];
    }}
}}
)",
                N);
            auto parameters = std::vector<Parameter> {
                { GetBuffer(), BufferSize() },
                { other.GetBuffer(), other.BufferSize() },
                { output.GetBuffer(), output.BufferSize() },
            };
            adapter->Execute(code, { parameters.begin(), parameters.end() }, N, 256);
        }

        return output;
    }

    GpuMatrix Sigmoid() const
    {
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = GpuMatrix { m_row, m_column };

        // Caculate mat4x4
        size_t N = (m_paddingRow >> 2) * m_paddingColumn;
        auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {}) {{
        output[i] = 1 / (1 + exp(-input[i]));
    }}
}}
)",
            N);
        auto parameters = std::vector<Parameter> {
            { GetBuffer(), BufferSize() },
            { output.GetBuffer(), output.BufferSize() },
        };
        adapter->Execute(code, { parameters.begin(), parameters.end() }, N, 256);
        return output;
    }

    GpuMatrix Transpose() const
    {
        auto output = GpuMatrix { m_column, m_row };

        // Caculate mat4x4
        size_t N = (m_paddingRow >> 2) * (m_paddingColumn >> 2);
        auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input: array<mat4x4f>;
@group(0) @binding(1) var<storage, read_write> output: array<mat4x4f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {0}) {{
        output[(i % {2}) * {1} + (i / {2})] = transpose(input[i]);
    }}
}}
)",
            N, m_paddingRow >> 2, m_paddingColumn >> 2);
        auto parameters = std::vector<Parameter> {
            { GetBuffer(), BufferSize() },
            { output.GetBuffer(), output.BufferSize() },
        };
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        adapter->Execute(code, { parameters.begin(), parameters.end() }, N, 256);
        return output;
    }

    GpuMatrix ElementProduct(const GpuMatrix& other)
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = GpuMatrix { m_row, m_column };

        // Caculate vec4x4
        size_t N = (m_paddingRow >> 2) * m_paddingColumn;
        if (N) {
            auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> input2: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {}) {{
        output[i] = input1[i] * input2[i];
    }}
}}
)",
                N);
            auto parameters = std::vector<Parameter> {
                { GetBuffer(), BufferSize() },
                { other.GetBuffer(), other.BufferSize() },
                { output.GetBuffer(), output.BufferSize() },
            };
            adapter->Execute(code, { parameters.begin(), parameters.end() }, N, 256);
        }

        return output;
    }

    std::vector<float> Read() const
    {
        std::vector<float> out(m_row * m_column);
        MapBuffer([this, &out](const float* data) {
            for (auto y = 0u; y < m_row; ++y) {
                for (auto x = 0u; x < m_column; ++x) {
                    auto i = IndexInMat4x4ArrayMemory(y, x);
                    out.data()[y * m_column + x] = data[i];
                }
            }
        });
        return out;
    }

    float operator[](size_t row, size_t column) const
    {
        if (row >= m_row || column >= m_column) {
            throw std::runtime_error { "Out of range" };
        }

        float ret {};
        MapBuffer([this, row, column, &ret](const float* data) {
            auto i = IndexInMat4x4ArrayMemory(row, column);
            ret = data[i];
        });
        return ret;
    }

private:
    int IndexInMat4x4ArrayMemory(int row, int column) const
    {
        return (((row >> 2) * (m_paddingColumn >> 2) + (column >> 2)) << 4) + (((row & 0x3) << 2) + (column & 0x3));
    }

    void MapBuffer(std::function<void(const float*)> callback) const
    {
        auto adapter = GpuInstance::GetInstance().GetAdapter();

        auto bufferSize = BufferSize();

        auto readbackBufferDescriptor = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
            .size = bufferSize,
        };

        auto pReadbackBuffer = gpu_ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> { wgpuDeviceCreateBuffer(adapter->GetDevice(), &readbackBufferDescriptor) };

        auto commandEncoder = wgpuDeviceCreateCommandEncoder(adapter->GetDevice(), nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, m_pBuffer.get(), 0, pReadbackBuffer.get(), 0, bufferSize);
        auto commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);

        auto submitPromise = std::promise<WGPUQueueWorkDoneStatus>();
        auto submitFuture = submitPromise.get_future();
        wgpuQueueSubmit(adapter->GetQueue(), 1, &commandBuffer);
        wgpuQueueOnSubmittedWorkDone(adapter->GetQueue(), [](WGPUQueueWorkDoneStatus status, void* callbackData) { ((std::promise<WGPUQueueWorkDoneStatus>*)callbackData)->set_value(status); }, &submitPromise);
        if (auto status = Wait(submitFuture); status != WGPUQueueWorkDoneStatus_Success) {
            throw std::runtime_error { "wgpuQueueOnSubmittedWorkDone failed." };
        }

        auto mapPromise = std::promise<WGPUBufferMapAsyncStatus>();
        auto mapFuture = mapPromise.get_future();
        wgpuBufferMapAsync(pReadbackBuffer.get(), WGPUMapMode_Read, 0, bufferSize, [](WGPUBufferMapAsyncStatus status, void* captureData) { ((std::promise<WGPUBufferMapAsyncStatus>*)captureData)->set_value(status); }, &mapPromise);
        if (auto status = Wait(mapFuture); status != WGPUBufferMapAsyncStatus_Success) {
            throw std::runtime_error { "wgpuBufferMapAsync failed." };
        }

        const auto* pMappedData = (float*)wgpuBufferGetConstMappedRange(pReadbackBuffer.get(), 0, bufferSize);
        callback(pMappedData);
        wgpuBufferUnmap(pReadbackBuffer.get());
    }

    template <typename T>
    T Wait(std::future<T>& future) const
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

export GpuMatrix operator-(float v, const GpuMatrix& m)
{
    auto adapter = GpuInstance::GetInstance().GetAdapter();
    auto vbuffer = adapter->CreateBuffer(1);
    wgpuQueueWriteBuffer(adapter->GetQueue(), vbuffer.get(), 0, &v, sizeof(float));

    auto output = GpuMatrix { m.m_row, m.m_column };

    // Caculate mat4x4
    size_t N = (m.m_paddingRow >> 2) * m.m_paddingColumn;
    auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: f32;
@group(0) @binding(1) var<storage, read_write> input2: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {}) {{
        output[i] = input1 - input2[i];
    }}
}}
)",
        N);
    auto parameters = std::vector<Parameter> {
        { vbuffer.get(), sizeof(float) },
        { m.GetBuffer(), m.BufferSize() },
        { output.GetBuffer(), output.BufferSize() },
    };
    adapter->Execute(code, { parameters.begin(), parameters.end() }, N, 256);
    return output;
}

export GpuMatrix operator*(float v, const GpuMatrix& m)
{
    auto adapter = GpuInstance::GetInstance().GetAdapter();
    auto vbuffer = adapter->CreateBuffer(1);
    wgpuQueueWriteBuffer(adapter->GetQueue(), vbuffer.get(), 0, &v, sizeof(float));

    auto output = GpuMatrix { m.m_row, m.m_column };

    // Caculate mat4x4
    size_t N = (m.m_paddingRow >> 2) * m.m_paddingColumn;
    auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: f32;
@group(0) @binding(1) var<storage, read_write> input2: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i: u32 = global_id.x;
    if (i < {}) {{
        output[i] = input1 * input2[i];
    }}
}}
)",
        N);
    auto parameters = std::vector<Parameter> {
        { vbuffer.get(), sizeof(float) },
        { m.GetBuffer(), m.BufferSize() },
        { output.GetBuffer(), output.BufferSize() },
    };
    adapter->Execute(code, { parameters.begin(), parameters.end() }, N, 256);
    return output;
}
}