module;

#include <cstddef>

export module cpp_matrix:ref_ptr;

namespace cpp_matrix {

export template <typename TGPUNativeHandle, void (*GPUReference)(TGPUNativeHandle), void (*GPURelease)(TGPUNativeHandle)>
class ref_ptr {
public:
    ref_ptr() = default;

    ref_ptr(const ref_ptr& obj)
    {
        if ((m_handle = obj.m_handle)) {
            GPUReference(m_handle);
        }
    }

    ref_ptr(ref_ptr&& obj)
    {
        m_handle = obj.m_handle;
        obj.m_handle = nullptr;
    }

    explicit ref_ptr(TGPUNativeHandle handle)
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

    ref_ptr& operator=(ref_ptr&& r)
    {
        if (m_handle != r.m_handle) {
            *this = nullptr;
            m_handle = r.m_handle;
            r.m_handle = nullptr;
        }
        return *this;
    }

    ref_ptr& operator=(std::nullptr_t)
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

    const TGPUNativeHandle* get_addr() const
    {
        return &m_handle;
    }

protected:
    TGPUNativeHandle m_handle {};
};

}