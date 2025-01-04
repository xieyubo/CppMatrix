module;

#include <future>
#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <webgpu/webgpu.h>

export module webgpu:adapter;
import :gpu_ref_ptr;

namespace webgpu {

void ProcessGpuInstanceEvents();

export class GpuAdapter {
public:
    GpuAdapter() = default;

    GpuAdapter(WGPUAdapter adapter, WGPUDevice device)
        : m_pAdapter { std::move(adapter) }
        , m_pDevice { std::move(device) }
    {
        m_pQueue.reset(wgpuDeviceGetQueue(m_pDevice.get()));

        WGPUSupportedLimits supportedLimits {};
        if (wgpuAdapterGetLimits(m_pAdapter.get(), &supportedLimits) != WGPUStatus_Success) {
            throw std::runtime_error { "wgpuAdapterGetLimits failed." };
        }
        m_limits = supportedLimits.limits;
        m_isFloat16Supported = wgpuDeviceHasFeature(m_pDevice.get(), WGPUFeatureName_ShaderF16);
    }

    template <typename T>
    gpu_ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> CreateBuffer(size_t elementSize);

    template <>
    gpu_ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> CreateBuffer<_Float16>(size_t elementSize)
    {
        if (!m_isFloat16Supported) {
            throw std::runtime_error { "float16 is not supported." };
        }

        return DoCreateBuffer(sizeof(_Float16) * elementSize);
    }

    template <>
    gpu_ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> CreateBuffer<float>(size_t elementSize)
    {
        return DoCreateBuffer(sizeof(float) * elementSize);
    }

    template <typename T>
    gpu_ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> CreateBuffer(size_t row, size_t column)
    {
        return CreateBuffer<T>(row * column);
    }

    WGPUDevice GetDevice() const
    {
        return m_pDevice.get();
    }

    WGPUQueue GetQueue() const
    {
        return m_pQueue.get();
    }

    void Execute(std::string_view shaderScript, std::span<Parameter> parameters, size_t N, size_t batchSize)
    {
        auto hash = std::hash<std::string_view> {}(shaderScript);
        auto it = m_cachedShaderModules.find(hash);
        if (it == m_cachedShaderModules.end()) {
            it = m_cachedShaderModules.emplace(hash, BuildShaderModule(shaderScript)).first;
        }
        Execute(it->second.get(), { parameters.begin(), parameters.end() }, N, batchSize);
    }

private:
    GpuShaderModulePtr BuildShaderModule(std::string_view shaderScript)
    {
        // Create wgsl
        auto wgslDesc = WGPU_SHADER_SOURCE_WGSL_INIT;
        wgslDesc.code.data = shaderScript.data();
        wgslDesc.code.length = shaderScript.length();

        auto shaderModuleDesc = WGPUShaderModuleDescriptor {
            .nextInChain = &wgslDesc.chain,
        };

        return GpuShaderModulePtr { wgpuDeviceCreateShaderModule(m_pDevice.get(), &shaderModuleDesc) };
    }

    void Execute(WGPUShaderModule shaderModule, std::span<Parameter> parameters, size_t N, size_t batchSize)
    {
        // Create layout entries for parameters.
        auto layoutEntries = std::vector<WGPUBindGroupLayoutEntry>(parameters.size());
        for (auto i = 0u; i < parameters.size(); ++i) {
            layoutEntries[i] = WGPUBindGroupLayoutEntry {
                .binding = i,
                .visibility = WGPUShaderStage_Compute,
                .buffer = WGPUBufferBindingLayout {
                    .type = WGPUBufferBindingType_Storage,
                    .minBindingSize = parameters[i].size,
                },
            };
        }

        auto layoutDesc = WGPUBindGroupLayoutDescriptor {
            .entryCount = layoutEntries.size(),
            .entries = layoutEntries.data(),
        };

        auto layout = gpu_ref_ptr<WGPUBindGroupLayout, wgpuBindGroupLayoutAddRef, wgpuBindGroupLayoutRelease> {
            wgpuDeviceCreateBindGroupLayout(m_pDevice.get(), &layoutDesc)
        };

        // Create bind group entries.
        auto bindGroupEntries = std::vector<WGPUBindGroupEntry>(parameters.size());
        for (auto i = 0u; i < parameters.size(); ++i) {
            bindGroupEntries[i] = WGPUBindGroupEntry {
                .binding = i,
                .buffer = parameters[i].buffer,
                .size = parameters[i].size,
            };
        }

        auto bindGroupDesc = WGPUBindGroupDescriptor {
            .layout = layout.get(),
            .entryCount = bindGroupEntries.size(),
            .entries = bindGroupEntries.data(),
        };

        auto bindGroup = gpu_ref_ptr<WGPUBindGroup, wgpuBindGroupAddRef, wgpuBindGroupRelease> {
            wgpuDeviceCreateBindGroup(m_pDevice.get(), &bindGroupDesc)
        };

        // Create pipeline.
        auto pipelineLayoutDesc = WGPUPipelineLayoutDescriptor {
            .bindGroupLayoutCount = 1,
            .bindGroupLayouts = layout.get_addr(),
        };

        auto pipelineLayout = gpu_ref_ptr<WGPUPipelineLayout, wgpuPipelineLayoutAddRef, wgpuPipelineLayoutRelease> {
            wgpuDeviceCreatePipelineLayout(m_pDevice.get(), &pipelineLayoutDesc)
        };

        // Create wgsl pipeline.
        auto computePipelineDesc = WGPUComputePipelineDescriptor {
            .layout = pipelineLayout.get(),
            .compute = {
                .module = shaderModule,
                .entryPoint = {
                    .data = "main",
                    .length = 4 },
            },
        };

        auto computePipeline = gpu_ref_ptr<WGPUComputePipeline, wgpuComputePipelineAddRef, wgpuComputePipelineRelease> {
            wgpuDeviceCreateComputePipeline(m_pDevice.get(), &computePipelineDesc)
        };

        // reset command buffer.
        auto commandEncoder = gpu_ref_ptr<WGPUCommandEncoder, wgpuCommandEncoderAddRef, wgpuCommandEncoderRelease> {
            wgpuDeviceCreateCommandEncoder(m_pDevice.get(), nullptr)
        };
        auto computePassEncoder
            = gpu_ref_ptr<WGPUComputePassEncoder, wgpuComputePassEncoderAddRef, wgpuComputePassEncoderRelease> {
                  wgpuCommandEncoderBeginComputePass(commandEncoder.get(), nullptr)
              };
        wgpuComputePassEncoderSetPipeline(computePassEncoder.get(), computePipeline.get());
        wgpuComputePassEncoderSetBindGroup(computePassEncoder.get(), 0, bindGroup.get(), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder.get(), (N + (batchSize - 1)) / batchSize, 1, 1);
        wgpuComputePassEncoderEnd(computePassEncoder.get());

        auto commandBuffer = gpu_ref_ptr<WGPUCommandBuffer, wgpuCommandBufferAddRef, wgpuCommandBufferRelease> {
            wgpuCommandEncoderFinish(commandEncoder.get(), nullptr)
        };

        auto compilationPromise = std::promise<void> {};
        auto compilationFuture = compilationPromise.get_future();
        wgpuShaderModuleGetCompilationInfo(
            computePipelineDesc.compute.module,
            [](WGPUCompilationInfoRequestStatus status, WGPUCompilationInfo const* compilationInfo, void* userData) {
                if (compilationInfo) {
                    for (uint32_t i = 0; i < compilationInfo->messageCount; ++i) {
                        printf("Message %d: %s\n", i,
                            std::string {
                                compilationInfo->messages[i].message.data, compilationInfo->messages[i].message.length }
                                .c_str());
                    }
                    ((std::promise<void>*)userData)->set_value();
                }
            },
            &compilationPromise);
        Wait(compilationFuture);

        // Submit the command buffer.
        auto submitPromise = std::promise<void> {};
        auto submitFuture = submitPromise.get_future();
        wgpuQueueSubmit(m_pQueue.get(), 1, commandBuffer.get_addr());
        wgpuQueueOnSubmittedWorkDone(
            m_pQueue.get(),
            [](WGPUQueueWorkDoneStatus status, void* data) { ((std::promise<void>*)data)->set_value(); },
            &submitPromise);
        Wait(submitFuture);
    }

    gpu_ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> DoCreateBuffer(size_t byteSize)
    {
        // buffer need 4 bytes align.
        byteSize = (byteSize + 3) & ~3;
        if (byteSize > m_limits.maxStorageBufferBindingSize || byteSize > m_limits.maxBufferSize) {
            throw std::runtime_error { std::format("Buffer size is too big (required size is {}, limits is "
                                                   "maxStorageBufferBindingSize: {}, maxBufferSize: {}).",
                byteSize, m_limits.maxStorageBufferBindingSize, m_limits.maxBufferSize) };
        }

        auto bufferDesc = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
            .size = byteSize,
        };

        return gpu_ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> { wgpuDeviceCreateBuffer(
            m_pDevice.get(), &bufferDesc) };
    }

    template <typename T>
    T Wait(std::future<T>& future)
    {
        while (future.wait_for(std::chrono::milliseconds {}) != std::future_status::ready) {
            ProcessGpuInstanceEvents();
        }
        return future.get();
    }

    gpu_ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease> m_pAdapter {};
    gpu_ref_ptr<WGPUDevice, wgpuDeviceAddRef, wgpuDeviceRelease> m_pDevice {};
    gpu_ref_ptr<WGPUQueue, wgpuQueueAddRef, wgpuQueueRelease> m_pQueue {};
    WGPULimits m_limits {};
    bool m_isFloat16Supported {};
    std::unordered_map<size_t, GpuShaderModulePtr> m_cachedShaderModules {};
};

}