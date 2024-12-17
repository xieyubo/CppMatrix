module;

#include <cassert>
#include <coroutine>
#include <format>
#include <span>
#include <stdexcept>
#include <vector>
#include <webgpu/webgpu.h>

export module cpp_matrix:gpu_matrix;
import :ref_ptr;
import :gpu_instance;

namespace cpp_matrix {

export class GpuMatrix {
public:
    static bool IsSupported(size_t row, size_t column)
    {
        const auto& limits = GpuInstance::GetInstance().GetAdapter().GetLimits().limits;
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

        m_paddingRow = m_row == 3 && m_column != 1 ? 4 : m_row;
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        m_pBuffer = adapter.CreateBuffer(m_paddingRow, m_column);
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
        return sizeof(float) * m_paddingRow * m_column;
    }

    GpuMatrix& operator=(std::vector<float> data)
    {
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        m_pBuffer = adapter.CreateBuffer(m_paddingRow = m_row = 1, m_column = data.size());
        wgpuQueueWriteBuffer(adapter.GetQueue(), m_pBuffer.get(), 0, data.data(), sizeof(float) * data.size());
        return *this;
    }

    template <size_t N>
    void Write(std::span<float, N> data)
    {
        auto adapter = GpuInstance::GetInstance().GetAdapter();

        if (m_row == 1 || m_column == 1) {
            wgpuQueueWriteBuffer(adapter.GetQueue(), m_pBuffer.get(), 0, data.data(), data.size_bytes());
        } else {
            std::vector<float> tmp(m_paddingRow * m_column);
            for (auto y = 0u; y < m_row; ++y) {
                for (auto x = 0u; x < m_column; ++x) {
                    tmp.data()[x * m_paddingRow + y] = data.data()[y * m_column + x];
                }
            }
            wgpuQueueWriteBuffer(adapter.GetQueue(), m_pBuffer.get(), 0, tmp.data(), sizeof(float) * tmp.size());
        }
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

        if (m_row > 4 || m_column > 4) {
            throw std::runtime_error { "Shape is not supported." };
        }

        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = GpuMatrix { m_row, other.m_column };
        auto k = std::vector<Parameter> {
            { GetBuffer(), BufferSize() },
            { other.GetBuffer(), other.BufferSize() },
            { output.GetBuffer(), output.BufferSize() },
        };
        std::string code;
        if (m_row != 1 && m_column == 1 && other.m_column != 1) {
            if (other.m_column == 2) {
                code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: {};
@group(0) @binding(1) var<storage, read_write> input2: {};
@group(0) @binding(2) var<storage, read_write> output: {};
@compute @workgroup_size(1)
fn main() {{
    output = {}(input1 * input2[0], input1 * input2[1]);
}}
)",
                    GetWgslType(), other.GetWgslType(), output.GetWgslType(), output.GetWgslType());
            } else if (other.m_column == 3) {
                code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: {};
@group(0) @binding(1) var<storage, read_write> input2: {};
@group(0) @binding(2) var<storage, read_write> output: {};
@compute @workgroup_size(1)
fn main() {{
    output = {}(input1 * input2[0], input1 * input2[1], input1 * input2[2]);
}}
)",
                    GetWgslType(), other.GetWgslType(), output.GetWgslType(), output.GetWgslType());
            } else {
                assert(other.m_column == 4);
                code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: {};
@group(0) @binding(1) var<storage, read_write> input2: {};
@group(0) @binding(2) var<storage, read_write> output: {};
@compute @workgroup_size(1)
fn main() {{
    output = {}(input1 * input2[0], input1 * input2[1], input1 * input2[2], input1 * input2[3]);
}}
)",
                    GetWgslType(), other.GetWgslType(), output.GetWgslType(), output.GetWgslType());
            }
        } else if ((m_row == 1 && m_column != 1 || m_row != 1 && m_column == 1) && (other.m_row == 1 && other.m_column != 1 || other.m_row != 1 && other.m_column == 1)) {
            code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: {};
@group(0) @binding(1) var<storage, read_write> input2: {};
@group(0) @binding(2) var<storage, read_write> output: {};
@compute @workgroup_size(1)
fn main() {{
    output = dot(input1, input2);
}}
)",
                GetWgslType(), other.GetWgslType(), output.GetWgslType());
        } else {
            code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: {};
@group(0) @binding(1) var<storage, read_write> input2: {};
@group(0) @binding(2) var<storage, read_write> output: {};
@compute @workgroup_size(1)
fn main() {{
    output = input1 * input2;
}}
)",
                GetWgslType(), other.GetWgslType(), output.GetWgslType());
        }
        adapter.Run(code.c_str(), { k.begin(), k.end() }, 1).await_resume();
        return output;
    }

    GpuMatrix operator+(const GpuMatrix& other) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = GpuMatrix { m_row, m_column };

        // Caculate vec4f, batch size = 256.
        size_t total = m_paddingRow * m_column;
        size_t N = total / 4;
        if (N) {
            auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> input2: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
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
            adapter.Run(code.c_str(), { parameters.begin(), parameters.end() }, N, 256);
            total -= N * 4;
        }

        if (total) {
            auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: {0};
@group(0) @binding(1) var<storage, read_write> input2: {0};
@group(0) @binding(2) var<storage, read_write> output: {0};
@compute @workgroup_size(1)
fn main() {{
    output = input1 + input2;
}}
)",
                total == 1 ? "f32" : std::format("vec{}f", total));
            auto parameters = std::vector<Parameter> {
                { GetBuffer(), BufferSize() - sizeof(float) * 4 * N, sizeof(float) * 4 * N },
                { other.GetBuffer(), BufferSize() - sizeof(float) * 4 * N, sizeof(float) * 4 * N },
                { output.GetBuffer(), BufferSize() - sizeof(float) * 4 * N, sizeof(float) * 4 * N },
            };
            adapter.Run(code.c_str(), { parameters.begin(), parameters.end() });
        }
        return output;
    }

    std::vector<float> Read() const
    {
        return ReadAsync().await_resume();
    }

    float operator[](size_t row, size_t column) const
    {
        if (row >= m_row || column >= m_column) {
            throw std::runtime_error { "Out of range" };
        }

        auto adapter = GpuInstance::GetInstance().GetAdapter();

        auto bufferSize = BufferSize();

        auto readbackBufferDescriptor = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
            .size = bufferSize,
        };

        auto pReadbackBuffer = ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> { wgpuDeviceCreateBuffer(adapter.GetDevice(), &readbackBufferDescriptor) };

        auto commandEncoder = wgpuDeviceCreateCommandEncoder(adapter.GetDevice(), nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, m_pBuffer.get(), 0, pReadbackBuffer.get(), 0, bufferSize);
        auto commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);

        auto submitPromise = Promise<void>();
        wgpuQueueSubmit(adapter.GetQueue(), 1, &commandBuffer);
        wgpuQueueOnSubmittedWorkDone(adapter.GetQueue(), [](WGPUQueueWorkDoneStatus status, void* callbackData) { (*Promise<void>::GetState(callbackData))->SetValue(); }, submitPromise.GetState().release());
        submitPromise.await_resume();

        auto mapPromise = Promise<void>();
        wgpuBufferMapAsync(pReadbackBuffer.get(), WGPUMapMode_Read, 0, bufferSize, [](WGPUBufferMapAsyncStatus status, void* captureData) { (*Promise<void>::GetState(captureData))->SetValue(); }, mapPromise.GetState().release());
        mapPromise.await_resume();

        const auto* pMappedData = (float*)wgpuBufferGetConstMappedRange(pReadbackBuffer.get(), 0, bufferSize);
        auto res = pMappedData[column * m_paddingRow + row];
        wgpuBufferUnmap(pReadbackBuffer.get());
        return res;
    }

private:
    Promise<std::vector<float>> ReadAsync() const
    {
        auto adapter = GpuInstance::GetInstance().GetAdapter();

        auto bufferSize = BufferSize();

        auto readbackBufferDescriptor = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
            .size = bufferSize,
        };

        auto pReadbackBuffer = ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> { wgpuDeviceCreateBuffer(adapter.GetDevice(), &readbackBufferDescriptor) };

        auto commandEncoder = wgpuDeviceCreateCommandEncoder(adapter.GetDevice(), nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, m_pBuffer.get(), 0, pReadbackBuffer.get(), 0, bufferSize);
        auto commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);

        auto submitPromise = Promise<WGPUQueueWorkDoneStatus>();
        wgpuQueueSubmit(adapter.GetQueue(), 1, &commandBuffer);
        wgpuQueueOnSubmittedWorkDone(adapter.GetQueue(), [](WGPUQueueWorkDoneStatus status, void* callbackData) { (*Promise<WGPUQueueWorkDoneStatus>::GetState(callbackData))->SetValue(status); }, submitPromise.GetState().release());
        if (auto status = co_await submitPromise; status != WGPUQueueWorkDoneStatus_Success) {
            throw std::runtime_error { "wgpuQueueOnSubmittedWorkDone failed." };
        }

        auto mapPromise = Promise<WGPUBufferMapAsyncStatus>();
        wgpuBufferMapAsync(pReadbackBuffer.get(), WGPUMapMode_Read, 0, bufferSize, [](WGPUBufferMapAsyncStatus status, void* captureData) { (*Promise<WGPUBufferMapAsyncStatus>::GetState(captureData))->SetValue(status); }, mapPromise.GetState().release());
        if (auto status = co_await mapPromise; status != WGPUBufferMapAsyncStatus_Success) {
            throw std::runtime_error { "wgpuBufferMapAsync failed." };
        }

        const auto* pMappedData = (float*)wgpuBufferGetConstMappedRange(pReadbackBuffer.get(), 0, bufferSize);
        std::vector<float> out(m_row * m_column);
        for (auto y = 0u; y < m_row; ++y) {
            for (auto x = 0u; x < m_column; ++x) {
                out.data()[y * m_column + x] = pMappedData[x * m_paddingRow + y];
            }
        }
        wgpuBufferUnmap(pReadbackBuffer.get());
        co_return out;
    }

    std::string GetWgslType() const
    {
        if (m_row == 1 && m_column == 1) {
            return "f32";
        } else if (m_row == 1 || m_column == 1) {
            return std::format("vec{}f", m_row * m_column);
        } else {
            return std::format("mat{}x{}f", m_column, m_row);
        }
    }

    size_t m_row {};
    size_t m_paddingRow {};
    size_t m_column {};
    ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> m_pBuffer {};
};

}