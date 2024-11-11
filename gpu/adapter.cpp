module;

#include <webgpu/webgpu.h>

export module gpu:adapter;
import :gpu_ref_ptr;

namespace gpu {

export class Adapter {
public:
    Adapter() = default;

    Adapter(WGPUAdapter adapter)
        : m_pAdapter { adapter }
    {
    }

private:
    gpu_ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease> m_pAdapter {};
};

}