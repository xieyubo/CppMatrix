module;

#include <future>
#include <span>
#include <string_view>
#include <vector>
#include <webgpu/webgpu.h>

export module cpp_matrix:gpu_compute_core;
import :gpu_ref_ptr;

namespace cpp_matrix {

export class GpuComputeCore {
public:
    GpuComputeCore() = default;
    
    GpuComputeCore(const GpuComputeCore&) = default;

    GpuComputeCore(WGPUDevice device, std::string_view shaderScript)
        : m_device { device }
        , m_adapter { wgpuDeviceGetAdapter(device) }
        , m_queue { wgpuDeviceGetQueue(device) }
    {
        // Create wgsl
        auto wgslDesc = WGPU_SHADER_SOURCE_WGSL_INIT;
        wgslDesc.code.data = shaderScript.data();
        wgslDesc.code.length = shaderScript.length();

        auto shaderModuleDesc = WGPUShaderModuleDescriptor {
            .nextInChain = &wgslDesc.chain,
        };

        m_shaderModule = GpuShaderModule { wgpuDeviceCreateShaderModule(device, &shaderModuleDesc) };
    }

    GpuComputeCore& operator=(const GpuComputeCore&) = default;

    void Execute(std::span<Parameter> parameters, size_t N, size_t batchSize)
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

        auto layout = gpu_ref_ptr<WGPUBindGroupLayout, wgpuBindGroupLayoutAddRef, wgpuBindGroupLayoutRelease> { wgpuDeviceCreateBindGroupLayout(m_device.get(), &layoutDesc) };

        // Create bind group entries.
        auto bindGroupEntries = std::vector<WGPUBindGroupEntry>(parameters.size());
        for (auto i = 0u; i < parameters.size(); ++i) {
            bindGroupEntries[i] = WGPUBindGroupEntry {
                .binding = i,
                .buffer = parameters[i].buffer,
                .offset = parameters[i].offset,
                .size = parameters[i].size,
            };
        }

        auto bindGroupDesc = WGPUBindGroupDescriptor {
            .layout = layout.get(),
            .entryCount = bindGroupEntries.size(),
            .entries = bindGroupEntries.data(),
        };

        auto bindGroup = gpu_ref_ptr<WGPUBindGroup, wgpuBindGroupAddRef, wgpuBindGroupRelease> { wgpuDeviceCreateBindGroup(m_device.get(), &bindGroupDesc) };

        // Create pipeline.
        auto pipelineLayoutDesc = WGPUPipelineLayoutDescriptor {
            .bindGroupLayoutCount = 1,
            .bindGroupLayouts = layout.get_addr(),
        };

        auto pipelineLayout = gpu_ref_ptr<WGPUPipelineLayout, wgpuPipelineLayoutAddRef, wgpuPipelineLayoutRelease> { wgpuDeviceCreatePipelineLayout(m_device.get(), &pipelineLayoutDesc) };

        // Create wgsl pipeline.
        auto computePipelineDesc = WGPUComputePipelineDescriptor {
            .layout = pipelineLayout.get(),
            .compute = {
                .module = m_shaderModule.get(),
                .entryPoint = {
                    .data = "main",
                    .length = 4 },
            },
        };

        auto computePipeline = gpu_ref_ptr<WGPUComputePipeline, wgpuComputePipelineAddRef, wgpuComputePipelineRelease> { wgpuDeviceCreateComputePipeline(m_device.get(), &computePipelineDesc) };

        // reset command buffer.
        auto commandEncoder = gpu_ref_ptr<WGPUCommandEncoder, wgpuCommandEncoderAddRef, wgpuCommandEncoderRelease> { wgpuDeviceCreateCommandEncoder(m_device.get(), nullptr) };
        auto computePassEncoder = gpu_ref_ptr<WGPUComputePassEncoder, wgpuComputePassEncoderAddRef, wgpuComputePassEncoderRelease> { wgpuCommandEncoderBeginComputePass(commandEncoder.get(), nullptr) };
        wgpuComputePassEncoderSetPipeline(computePassEncoder.get(), computePipeline.get());
        wgpuComputePassEncoderSetBindGroup(computePassEncoder.get(), 0, bindGroup.get(), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder.get(), (N + (batchSize - 1)) / batchSize, 1, 1);
        wgpuComputePassEncoderEnd(computePassEncoder.get());

        auto commandBuffer = gpu_ref_ptr<WGPUCommandBuffer, wgpuCommandBufferAddRef, wgpuCommandBufferRelease> { wgpuCommandEncoderFinish(commandEncoder.get(), nullptr) };

        auto compilationPromise = std::promise<void> {};
        auto compilationFuture = compilationPromise.get_future();
        wgpuShaderModuleGetCompilationInfo(computePipelineDesc.compute.module, [](WGPUCompilationInfoRequestStatus status, WGPUCompilationInfo const* compilationInfo, void* userData) {
        if (compilationInfo) {
            for (uint32_t i = 0; i < compilationInfo->messageCount; ++i) {
                printf("Message %d: %s\n", i, std::string { compilationInfo->messages[i].message.data, compilationInfo->messages[i].message.length }.c_str());
            }
            ((std::promise<void>*)userData)->set_value();
        } }, &compilationPromise);
        Wait(compilationFuture);

        // Submit the command buffer.
        auto submitPromise = std::promise<void> {};
        auto submitFuture = submitPromise.get_future();
        wgpuQueueSubmit(m_queue, 1, commandBuffer.get_addr());
        wgpuQueueOnSubmittedWorkDone(m_queue, [](WGPUQueueWorkDoneStatus status, void* data) { ((std::promise<void>*)data)->set_value(); }, &submitPromise);
        Wait(submitFuture);
    }

    operator bool() const
    {
        return m_shaderModule;
    }

private:
    GpuDevicePtr m_device {};
    GpuAdapterPtr m_adapter {};
    GpuQueuePtr m_queue {};
    GpuShaderModule m_shaderModule {};
};

}