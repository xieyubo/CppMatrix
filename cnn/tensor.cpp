module;

#include <webgpu/webgpu.h>

export module cnn:tensor;
import :shape;

namespace cnn {

export struct Tensor {
    WGPUBuffer buffer {};
    Shape shape {};

    ~Tensor()
    {
        if (buffer) {
            // wgpuBufferRelease(buffer);
        }
    }

    size_t size_in_bytes() const {
        return sizeof(float) * shape.size();
    }
};

}