module;

#include <memory>
#include <span>
#include <utility>
#include <webgpu/webgpu.h>

export module cpp_matrix:adapter;
import :promise;
import :ref_ptr;

namespace cpp_matrix {

export class Matrix;

export class Adapter {
public:
    Adapter() = default;

    Adapter(WGPUAdapter adapter, WGPUDevice device)
        : m_pAdapter { std::move(adapter) }
        , m_pDevice { std::move(device) }
    {
        m_pQueue.reset(wgpuDeviceGetQueue(m_pDevice.get()));
    }

    ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> CreateBuffer(size_t row, size_t column)
    {
        auto bufferDesc = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
            .size = sizeof(float) * row * column,
        };

        return ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> { wgpuDeviceCreateBuffer(m_pDevice.get(), &bufferDesc) };
    }

    Matrix CreateMatrix(size_t row, size_t column);

    WGPUDevice GetDevice() const
    {
        return m_pDevice.get();
    }

    WGPUQueue GetQueue() const
    {
        return m_pQueue.get();
    }

    Promise<void> Run(const char* shaderScript, std::span<Matrix> buffers, size_t batchSize);

    operator bool() const
    {
        return m_pAdapter && m_pDevice && m_pQueue;
    }

private:
    ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease> m_pAdapter {};
    ref_ptr<WGPUDevice, wgpuDeviceAddRef, wgpuDeviceRelease> m_pDevice {};
    ref_ptr<WGPUQueue, wgpuQueueAddRef, wgpuQueueRelease> m_pQueue {};
};

}