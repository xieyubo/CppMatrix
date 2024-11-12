module;

#include <memory>
#include <span>
#include <utility>
#include <webgpu/webgpu.h>

export module cnn:adapter;
import :dimension;
import :promise;
import :ref_ptr;

namespace cnn {

export class Tensor;

export class Adapter {
public:
    Adapter() = default;

    Adapter(WGPUAdapter adapter, WGPUDevice device)
        : m_pAdapter { std::move(adapter) }
        , m_pDevice { std::move(device) }
    {
        m_pQueue.reset(wgpuDeviceGetQueue(m_pDevice.get()));
    }

    Tensor CreateBuffer(Dimension dimension);

    WGPUDevice GetDevice() const
    {
        return m_pDevice.get();
    }

    WGPUQueue GetQueue() const
    {
        return m_pQueue.get();
    }

    Promise<void> Run(const char* shaderScript, std::span<Tensor> buffers, size_t batchSize);

private:
    ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease> m_pAdapter {};
    ref_ptr<WGPUDevice, wgpuDeviceAddRef, wgpuDeviceRelease> m_pDevice {};
    ref_ptr<WGPUQueue, wgpuQueueAddRef, wgpuQueueRelease> m_pQueue {};
};

}