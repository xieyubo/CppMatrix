module;

#include <utility>
#include <webgpu/webgpu.h>

export module gpu:adapter;
import :buffer;
import :dimension;
import :ref_ptr;

namespace gpu {

export class Adapter {
public:
    Adapter() = default;

    Adapter(WGPUAdapter adapter, WGPUDevice device)
        : m_pAdapter { std::move(adapter) }
        , m_pDevice { std::move(device) }
    {
    }

    Buffer CreateBuffer(Dimension dimension)
    {
        auto bufferDesc = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
            .size = sizeof(float) * dimension.elements(),
        };

        return { std::move(dimension), wgpuDeviceCreateBuffer(m_pDevice.get(), &bufferDesc) };
    }

private:
    ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease> m_pAdapter {};
    ref_ptr<WGPUDevice, wgpuDeviceAddRef, wgpuDeviceRelease> m_pDevice {};
};

}