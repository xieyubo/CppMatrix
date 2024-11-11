module;

#include <utility>
#include <webgpu/webgpu.h>

export module gpu:buffer;
import :dimension;
import :ref_ptr;

namespace gpu {

export class Buffer {
public:
    Buffer(Dimension dimension, WGPUBuffer buffer)
        : m_dimension { std::move(dimension) }
        , m_pBuffer { std::move(buffer) }
    {
    }

private:
    Dimension m_dimension {};
    ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> m_pBuffer {};
};

}