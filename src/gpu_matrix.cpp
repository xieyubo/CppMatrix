module;

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
    GpuMatrix() = default;

    GpuMatrix(size_t row, size_t column)
        : m_row { row }
        , m_column { column }
    {
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

        // The max dimension supported by wgsl is 4x4, so if the matrix dimension is bigger than it,
        // we need split it.
        if (m_row <= 4 && m_column <= 4 && other.m_column <= 4) {
            // Perfect, no need split.
            return DotWithNoSplication(other).await_resume();
        } else {
            throw std::runtime_error { "not supported" };
        }
    }

    GpuMatrix operator+(const GpuMatrix& other) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        if (m_row > 4 || m_column > 4) {
            throw std::runtime_error { "Shape is not supported." };
        }

        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = GpuMatrix { m_row, m_column };
        auto k = std::vector<GpuMatrix> { *this, other, output };
        auto code = std::format(R"(
@group(0) @binding(0) var<storage, read_write> input1: {};
@group(0) @binding(1) var<storage, read_write> input2: {};
@group(0) @binding(2) var<storage, read_write> output: {};
@compute @workgroup_size(1)
fn main() {{
    output = input1 + input2;
}}
)",
            GetWgslType(), other.GetWgslType(), output.GetWgslType());
        adapter.Run(code.c_str(), { k.begin(), k.end() }, 1).await_resume();
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

        auto submitPromise = Promise<void>();
        wgpuQueueSubmit(adapter.GetQueue(), 1, &commandBuffer);
        wgpuQueueOnSubmittedWorkDone(adapter.GetQueue(), [](WGPUQueueWorkDoneStatus status, void* callbackData) { (*Promise<void>::GetState(callbackData))->SetValue(); }, submitPromise.GetState().release());
        co_await submitPromise;

        auto mapPromise = Promise<void>();
        wgpuBufferMapAsync(pReadbackBuffer.get(), WGPUMapMode_Read, 0, bufferSize, [](WGPUBufferMapAsyncStatus status, void* captureData) { (*Promise<void>::GetState(captureData))->SetValue(); }, mapPromise.GetState().release());
        co_await mapPromise;

        const auto* pMappedData = (float*)wgpuBufferGetConstMappedRange(pReadbackBuffer.get(), 0, bufferSize);
        std::vector<float> wwe { pMappedData, pMappedData + bufferSize };
        std::vector<float> out(m_row * m_column);
        for (auto y = 0u; y < m_row; ++y) {
            for (auto x = 0u; x < m_column; ++x) {
                out.data()[y * m_column + x] = pMappedData[x * m_paddingRow + y];
            }
        }
        wgpuBufferUnmap(pReadbackBuffer.get());
        co_return out;
    }

    Promise<GpuMatrix> DotWithNoSplication(const GpuMatrix& other) const
    {
        /*
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        auto output = GpuMatrix { m_row, other.m_column };
        auto k = std::vector<GpuMatrix> { *this, other, output };
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
        co_await adapter.Run(code.c_str(), { k.begin(), k.end() }, 1);
        co_return output;
        */
        return {};
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
    size_t m_paddingRow {};
    size_t m_column {};
    ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> m_pBuffer {};
};

}