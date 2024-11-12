module;

#include <memory>
#include <utility>
#include <webgpu/webgpu.h>

export module gpu:adapter;
import :dimension;
import :ref_ptr;

namespace gpu {

export class Buffer;

export class Adapter {
public:
    Adapter() = default;

    Adapter(WGPUAdapter adapter, WGPUDevice device)
        : m_pAdapter { std::move(adapter) }
        , m_pDevice { std::move(device) }
    {
        m_pQueue.reset(wgpuDeviceGetQueue(m_pDevice.get()));
    }

    Buffer CreateBuffer(Dimension dimension);

    WGPUDevice GetDevice() const
    {
        return m_pDevice.get();
    }

    WGPUQueue GetQueue() const
    {
        return m_pQueue.get();
    }

private:
    ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease> m_pAdapter {};
    ref_ptr<WGPUDevice, wgpuDeviceAddRef, wgpuDeviceRelease> m_pDevice {};
    ref_ptr<WGPUQueue, wgpuQueueAddRef, wgpuQueueRelease> m_pQueue {};
};

}