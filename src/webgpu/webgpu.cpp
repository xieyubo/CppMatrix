module;

#include <span>
#include <string_view>

export module webgpu:webgpu;
import :gpu_instance;
import :gpu_ref_ptr;

namespace webgpu {

export void Run(std::string_view shaderScript, std::span<Parameter> parameters, size_t N, size_t batchSize)
{
    GpuInstance::GetInstance().GetAdapter()->Execute(shaderScript, parameters, N, batchSize);
}

}