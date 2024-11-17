module;

#include <vector>
#include <webgpu/webgpu.h>

export module cpp_matrix:gpu_matrix;
import :ref_ptr;
import :gpu_instance;

namespace cpp_matrix {

export class GpuMatrix {
public:
    GpuMatrix() = default;

    size_t Row() const
    {
        return m_row;
    }

    size_t Column() const
    {
        return m_column;
    }

    GpuMatrix& operator=(float f)
    {
        auto adapter = GpuInstance::GetInstance().GetAdapter();
        m_pBuffer = adapter.CreateBuffer(m_row = 1, m_column = 1);
        wgpuQueueWriteBuffer(adapter.GetQueue(), m_pBuffer.get(), 0, &f, sizeof(f));
        return *this;
    }

private:
    size_t m_row {};
    size_t m_column {};
    ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> m_pBuffer {};
};

}