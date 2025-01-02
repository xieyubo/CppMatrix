module;

#include <cstddef>
#include <future>
#include <webgpu/webgpu.h>

export module webgpu:gpu_ref_ptr;

namespace webgpu {

export void ProcessGpuInstanceEvents();

export struct Parameter {
    WGPUBuffer buffer {};
    size_t size {};
};

export template <typename TGPUNativeHandle, void (*GPUReference)(TGPUNativeHandle), void (*GPURelease)(TGPUNativeHandle)>
class gpu_ref_ptr {
public:
    gpu_ref_ptr() = default;

    gpu_ref_ptr(const gpu_ref_ptr& obj)
    {
        if ((m_handle = obj.m_handle)) {
            GPUReference(m_handle);
        }
    }

    gpu_ref_ptr(gpu_ref_ptr&& obj)
    {
        m_handle = obj.m_handle;
        obj.m_handle = nullptr;
    }

    explicit gpu_ref_ptr(TGPUNativeHandle handle)
        : m_handle { handle }
    {
    }

    void reset(TGPUNativeHandle handle)
    {
        if (m_handle != handle) {
            *this = nullptr;
            m_handle = handle;
        }
    }

    TGPUNativeHandle release()
    {
        auto handle = m_handle;
        m_handle = {};
        return handle;
    }

    gpu_ref_ptr& operator=(const gpu_ref_ptr& r)
    {
        if (m_handle != r.m_handle) {
            *this = nullptr;
            if ((m_handle = r.m_handle)) {
                GPUReference(m_handle);
            }
        }
        return *this;
    }

    gpu_ref_ptr& operator=(gpu_ref_ptr&& r)
    {
        if (m_handle != r.m_handle) {
            *this = nullptr;
            m_handle = r.m_handle;
            r.m_handle = nullptr;
        }
        return *this;
    }

    gpu_ref_ptr& operator=(std::nullptr_t)
    {
        if (m_handle) {
            GPURelease(m_handle);
            m_handle = nullptr;
        }
        return *this;
    }

    operator bool() const
    {
        return m_handle;
    }

    operator TGPUNativeHandle() const
    {
        return m_handle;
    }

    TGPUNativeHandle get() const
    {
        return m_handle;
    }

    const TGPUNativeHandle* get_addr() const
    {
        return &m_handle;
    }

    const TGPUNativeHandle* operator&() const
    {
        return &m_handle;
    }

protected:
    TGPUNativeHandle m_handle {};
};

export using GpuDevicePtr = gpu_ref_ptr<WGPUDevice, wgpuDeviceAddRef, wgpuDeviceRelease>;
export using GpuCommandBufferPtr = gpu_ref_ptr<WGPUCommandBuffer, wgpuCommandBufferAddRef, wgpuCommandBufferRelease>;
export using GpuQueuePtr = gpu_ref_ptr<WGPUQueue, wgpuQueueAddRef, wgpuQueueRelease>;
export using GpuAdapterPtr = gpu_ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease>;
export using GpuShaderModule = gpu_ref_ptr<WGPUShaderModule, wgpuShaderModuleAddRef, wgpuShaderModuleRelease>;
export using GpuShaderModulePtr = gpu_ref_ptr<WGPUShaderModule, wgpuShaderModuleAddRef, wgpuShaderModuleRelease>;

template <typename T>
T Wait(std::future<T>& future)
{
    while (future.wait_for(std::chrono::milliseconds {}) != std::future_status::ready) {
        ProcessGpuInstanceEvents();
    }
    return future.get();
}

}