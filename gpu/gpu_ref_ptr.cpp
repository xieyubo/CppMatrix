module;

#include <cstddef>

export module gpu:gpu_ref_ptr;

namespace gpu {

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

    gpu_ref_ptr& Acquire(TGPUNativeHandle handle)
    {
        *this = nullptr;
        m_handle = handle;
        return *static_cast<gpu_ref_ptr*>(this);
    }

    TGPUNativeHandle Release()
    {
        auto handle = m_handle;
        m_handle = {};
        return handle;
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

    TGPUNativeHandle get() const
    {
        return m_handle;
    }

protected:
    TGPUNativeHandle m_handle {};
};

}