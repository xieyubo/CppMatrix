module;

#include <memory>
#include <webgpu/webgpu.h>

module gpu;
import :buffer;

namespace gpu {

Buffer Adapter::CreateBuffer(Dimension dimension)
{
    auto bufferDesc = WGPUBufferDescriptor {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
        .size = sizeof(float) * dimension.elements(),
    };

    return { std::move(dimension), *this, wgpuDeviceCreateBuffer(m_pDevice.get(), &bufferDesc) };
}

}