module;

#include <coroutine>
#include <memory>
#include <span>
#include <vector>
#include <webgpu/webgpu.h>

module cnn;
import :matrix;
import :promise;

namespace cnn {

Matrix Adapter::CreateMatrix(size_t row, size_t column)
{
    auto bufferDesc = WGPUBufferDescriptor {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
        .size = sizeof(float) * row * column,
    };

    return { row, column, *this, wgpuDeviceCreateBuffer(m_pDevice.get(), &bufferDesc) };
}

Promise<void> Adapter::Run(const char* shaderScript, std::span<Matrix> buffers, size_t batchSize)
{
    // Create layout entries for buffers.
    auto layoutEntries = std::vector<WGPUBindGroupLayoutEntry>(buffers.size());
    for (auto i = 0u; i < buffers.size(); ++i) {
        layoutEntries[i] = WGPUBindGroupLayoutEntry {
            .binding = i,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout {
                .type = WGPUBufferBindingType_Storage,
                .minBindingSize = buffers[i].SizeInBytes(),
            },
        };
    }

    auto layoutDesc = WGPUBindGroupLayoutDescriptor {
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };

    auto layout = wgpuDeviceCreateBindGroupLayout(m_pDevice.get(), &layoutDesc);

    // Create bind group entries.
    auto bindGroupEntries = std::vector<WGPUBindGroupEntry>(buffers.size());
    for (auto i = 0u; i < buffers.size(); ++i) {
        bindGroupEntries[i] = WGPUBindGroupEntry {
            .binding = i,
            .buffer = buffers[i].GetBuffer(),
            .size = buffers[i].SizeInBytes(),
        };
    }

    auto bindGroupDesc = WGPUBindGroupDescriptor {
        .layout = layout,
        .entryCount = bindGroupEntries.size(),
        .entries = bindGroupEntries.data(),
    };

    auto bindGroup = wgpuDeviceCreateBindGroup(m_pDevice.get(), &bindGroupDesc);

    // Create pipeline.
    auto pipelineLayoutDesc = WGPUPipelineLayoutDescriptor {
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &layout,
    };

    auto pipelineLayout = wgpuDeviceCreatePipelineLayout(m_pDevice.get(), &pipelineLayoutDesc);

    // Create wgsl
    auto wgslDesc = WGPUShaderModuleWGSLDescriptor {
        .chain = WGPUChainedStruct {
            .sType = WGPUSType_ShaderModuleWGSLDescriptor,
        },
        .code = shaderScript,
    };

    auto shaderModuleDesc = WGPUShaderModuleDescriptor {
        .nextInChain = &wgslDesc.chain,
        .label = "shader",
    };

    auto computePipelineDesc = WGPUComputePipelineDescriptor {
        .layout = pipelineLayout,
        .compute = WGPUProgrammableStageDescriptor {
            .module = wgpuDeviceCreateShaderModule(m_pDevice.get(), &shaderModuleDesc),
            .entryPoint = "main",
        },
    };

    auto computePipeline = wgpuDeviceCreateComputePipeline(m_pDevice.get(), &computePipelineDesc);

    // reset command buffer.
    auto commandEncoder = wgpuDeviceCreateCommandEncoder(m_pDevice.get(), nullptr);
    auto computePassEncoder = wgpuCommandEncoderBeginComputePass(commandEncoder, nullptr);
    wgpuComputePassEncoderSetPipeline(computePassEncoder, computePipeline);
    wgpuComputePassEncoderSetBindGroup(computePassEncoder, 0, bindGroup, 0, nullptr);
    // wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder, workgroupSize, 1, 1);
    wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder, batchSize, 1, 1);
    wgpuComputePassEncoderEnd(computePassEncoder);

    auto commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);

    auto compilationPromise = Promise<void> {};
    wgpuShaderModuleGetCompilationInfo(computePipelineDesc.compute.module, [](WGPUCompilationInfoRequestStatus status, WGPUCompilationInfo const* compilationInfo, void* userData) {
        if (compilationInfo) {
            for (uint32_t i = 0; i < compilationInfo->messageCount; ++i) {
                printf("Message %d: %s\n", i, compilationInfo->messages[i].message);
            }
            (*Promise<void>::GetState(userData))->SetValue();
        } }, compilationPromise.GetState().release());
    co_await compilationPromise;

    // Submit the command buffer.
    auto submitPromise = Promise<void> {};
    wgpuQueueSubmit(m_pQueue.get(), 1, &commandBuffer);
    wgpuQueueOnSubmittedWorkDone(m_pQueue.get(), [](WGPUQueueWorkDoneStatus status, void* data) { (*Promise<void>::GetState(data))->SetValue(); }, submitPromise.GetState().release());
    co_await submitPromise;
}

}