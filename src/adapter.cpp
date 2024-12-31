module;

#include <coroutine>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>
#include <webgpu/webgpu.h>

export module cpp_matrix:adapter;
import :promise;
import :ref_ptr;

namespace cpp_matrix {

export class Matrix;
export class GpuMatrix;

export struct Parameter {
    WGPUBuffer buffer {};
    size_t size {};
    size_t offset {};
};

export class Adapter {
public:
    Adapter() = default;

    Adapter(WGPUAdapter adapter, WGPUDevice device)
        : m_pAdapter { std::move(adapter) }
        , m_pDevice { std::move(device) }
    {
        m_pQueue.reset(wgpuDeviceGetQueue(m_pDevice.get()));

        if (wgpuAdapterGetLimits(m_pAdapter.get(), &m_limits) != WGPUStatus_Success) {
            throw std::runtime_error { "wgpuAdapterGetLimits failed." };
        }
    }

    ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> CreateBuffer(size_t elementSize)
    {
        auto bufferDesc = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
            .size = sizeof(float) * elementSize,
        };

        return ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> { wgpuDeviceCreateBuffer(m_pDevice.get(), &bufferDesc) };
    }

    ref_ptr<WGPUBuffer, wgpuBufferAddRef, wgpuBufferRelease> CreateBuffer(size_t row, size_t column)
    {
        return CreateBuffer(row * column);
    }

    const WGPUSupportedLimits& GetLimits() const
    {
        return m_limits;
    }

    WGPUDevice GetDevice() const
    {
        return m_pDevice.get();
    }

    WGPUQueue GetQueue() const
    {
        return m_pQueue.get();
    }

    Promise<void> Run(std::string_view shaderScript, std::span<Parameter> parameters)
    {
        return Run(shaderScript, parameters, /*batchSize=*/1);
    }

    Promise<void> Run(std::string_view shaderScript, std::span<Parameter> parameters, size_t batchSize)
    {
        return Run(shaderScript, parameters, /*N=*/batchSize, batchSize);
    }

    Promise<void> Run(std::string_view shaderScript, std::span<Parameter> parameters, size_t N, size_t batchSize)
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

        auto layout = ref_ptr<WGPUBindGroupLayout, wgpuBindGroupLayoutAddRef, wgpuBindGroupLayoutRelease> { wgpuDeviceCreateBindGroupLayout(m_pDevice.get(), &layoutDesc) };

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

        auto bindGroup = ref_ptr<WGPUBindGroup, wgpuBindGroupAddRef, wgpuBindGroupRelease> { wgpuDeviceCreateBindGroup(m_pDevice.get(), &bindGroupDesc) };

        // Create pipeline.
        auto pipelineLayoutDesc = WGPUPipelineLayoutDescriptor {
            .bindGroupLayoutCount = 1,
            .bindGroupLayouts = layout.get_addr(),
        };

        auto pipelineLayout = ref_ptr<WGPUPipelineLayout, wgpuPipelineLayoutAddRef, wgpuPipelineLayoutRelease> { wgpuDeviceCreatePipelineLayout(m_pDevice.get(), &pipelineLayoutDesc) };

        // Create wgsl
        auto wgslDesc = WGPU_SHADER_SOURCE_WGSL_INIT;
        wgslDesc.code.data = shaderScript.data();
        wgslDesc.code.length = shaderScript.length();

        auto shaderModuleDesc = WGPUShaderModuleDescriptor {
            .nextInChain = &wgslDesc.chain,
        };

        auto shaderModule = ref_ptr<WGPUShaderModule, wgpuShaderModuleAddRef, wgpuShaderModuleRelease> { wgpuDeviceCreateShaderModule(m_pDevice.get(), &shaderModuleDesc) };
        auto computePipelineDesc = WGPUComputePipelineDescriptor {
            .layout = pipelineLayout.get(),
            .compute = WGPUProgrammableStageDescriptor {
                .module = shaderModule.get(),
                .entryPoint = {
                    .data = "main",
                    .length = 4 },
            },
        };

        auto computePipeline = ref_ptr<WGPUComputePipeline, wgpuComputePipelineAddRef, wgpuComputePipelineRelease> { wgpuDeviceCreateComputePipeline(m_pDevice.get(), &computePipelineDesc) };

        // reset command buffer.
        auto commandEncoder = ref_ptr<WGPUCommandEncoder, wgpuCommandEncoderAddRef, wgpuCommandEncoderRelease> { wgpuDeviceCreateCommandEncoder(m_pDevice.get(), nullptr) };
        auto computePassEncoder = ref_ptr<WGPUComputePassEncoder, wgpuComputePassEncoderAddRef, wgpuComputePassEncoderRelease> { wgpuCommandEncoderBeginComputePass(commandEncoder.get(), nullptr) };
        wgpuComputePassEncoderSetPipeline(computePassEncoder.get(), computePipeline.get());
        wgpuComputePassEncoderSetBindGroup(computePassEncoder.get(), 0, bindGroup.get(), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder.get(), (N + (batchSize - 1)) / batchSize, 1, 1);
        wgpuComputePassEncoderEnd(computePassEncoder.get());

        auto commandBuffer = ref_ptr<WGPUCommandBuffer, wgpuCommandBufferAddRef, wgpuCommandBufferRelease> { wgpuCommandEncoderFinish(commandEncoder.get(), nullptr) };

        auto compilationPromise = Promise<void> {};
        wgpuShaderModuleGetCompilationInfo(computePipelineDesc.compute.module, [](WGPUCompilationInfoRequestStatus status, WGPUCompilationInfo const* compilationInfo, void* userData) {
        if (compilationInfo) {
            for (uint32_t i = 0; i < compilationInfo->messageCount; ++i) {
                printf("Message %d: %s\n", i, std::string { compilationInfo->messages[i].message.data, compilationInfo->messages[i].message.length }.c_str());
            }
            (*Promise<void>::GetState(userData))->SetValue();
        } }, compilationPromise.GetState().release());
        co_await compilationPromise;

        // Submit the command buffer.
        auto submitPromise = Promise<void> {};
        wgpuQueueSubmit(m_pQueue.get(), 1, commandBuffer.get_addr());
        wgpuQueueOnSubmittedWorkDone(m_pQueue.get(), [](WGPUQueueWorkDoneStatus status, void* data) { (*Promise<void>::GetState(data))->SetValue(); }, submitPromise.GetState().release());
        co_await submitPromise;
    }

private:
    ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease> m_pAdapter {};
    ref_ptr<WGPUDevice, wgpuDeviceAddRef, wgpuDeviceRelease> m_pDevice {};
    ref_ptr<WGPUQueue, wgpuQueueAddRef, wgpuQueueRelease> m_pQueue {};
    WGPUSupportedLimits m_limits {};
};

}