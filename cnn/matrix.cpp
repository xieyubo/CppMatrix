module;

#include <coroutine>
#include <span>
#include <utility>
#include <vector>
#include <webgpu/webgpu.h>

export module cnn:matrix;
import :adapter;
import :promise;
import :ref_ptr;

namespace cnn {

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

    Matrix operator*(const Matrix& other)
    {
        return {};
        //auto row = m_dimension.
    }

private:
    size_t m_row {};
    size_t m_column {};
    Adapter m_adapter {};
    ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> m_pBuffer {};
};

}