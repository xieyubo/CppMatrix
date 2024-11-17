module;

#include <coroutine>
#include <cstdlib>
#include <format>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <webgpu/webgpu.h>

export module gpu_matrix:matrix;
import :adapter;
import :promise;
import :ref_ptr;

namespace gpu_matrix {

export class Matrix {
public:
    Matrix() = default;

    Matrix(size_t row, size_t column, Adapter adapter, WGPUBuffer buffer)
        : m_row { row }
        , m_column { column }
        , m_adapter { std::move(adapter) }
        , m_pBuffer { std::move(buffer) }
    {
    }

    template <size_t N>
    void Write(std::span<float, N> data)
    {
        wgpuQueueWriteBuffer(m_adapter.GetQueue(), m_pBuffer.get(), 0, data.data(), data.size_bytes());
    }

    Promise<std::vector<float>> Read()
    {
        auto bufferSize = sizeof(float) * m_row * m_column;

        auto readbackBufferDescriptor = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
            .size = bufferSize,
        };

        auto pReadbackBuffer = ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> { wgpuDeviceCreateBuffer(m_adapter.GetDevice(), &readbackBufferDescriptor) };

        auto commandEncoder = wgpuDeviceCreateCommandEncoder(m_adapter.GetDevice(), nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, m_pBuffer.get(), 0, pReadbackBuffer.get(), 0, bufferSize);
        auto commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);

        auto submitPromise = Promise<void>();
        wgpuQueueSubmit(m_adapter.GetQueue(), 1, &commandBuffer);
        wgpuQueueOnSubmittedWorkDone(m_adapter.GetQueue(), [](WGPUQueueWorkDoneStatus status, void* callbackData) { (*Promise<void>::GetState(callbackData))->SetValue(); }, submitPromise.GetState().release());
        co_await submitPromise;

        auto mapPromise = Promise<void>();
        wgpuBufferMapAsync(pReadbackBuffer.get(), WGPUMapMode_Read, 0, bufferSize, [](WGPUBufferMapAsyncStatus status, void* captureData) { (*Promise<void>::GetState(captureData))->SetValue(); }, mapPromise.GetState().release());
        co_await mapPromise;

        const auto* pMappedData = (float*)wgpuBufferGetConstMappedRange(pReadbackBuffer.get(), 0, bufferSize);
        std::vector<float> out { pMappedData, pMappedData + m_row * m_column };
        wgpuBufferUnmap(pReadbackBuffer.get());
        co_return out;
    }

    size_t SizeInBytes() const
    {
        return sizeof(float) * m_row * m_column;
    }

    WGPUBuffer GetBuffer() const
    {
        return m_pBuffer.get();
    }

    operator bool() const
    {
        return m_pBuffer;
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
        return m_row;
    }

    size_t Column() const
    {
        return m_column;
    }

private:
    Promise<Matrix> DotWithNoSplication(const Matrix& other)
    {
        auto output = m_adapter.CreateMatrix(m_row, other.m_column);
        auto k = std::vector<gpu_matrix::Matrix> { *this, other, output };
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
        if (m_row == 1 && m_column == 1)
        {
            return "f32";
        }
        else if (m_row == 1)
        {
            return std::format("array<f32, {}>", m_column);
        }
        else if (m_column == 1)
        {
            return std::format("vec{}f", m_row);
        }
        else
        {
            return std::format("mat{}x{}f", m_column, m_row);
        }
    }

    size_t m_row {};
    size_t m_column {};
    Adapter m_adapter {};
    ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> m_pBuffer {};
};

}