module;

#include <cstdio>
#include <future>
#include <span>
#include <thread>
#include <vector>
#include <webgpu/webgpu.h>

export module gpu:context;
import :shape;
import :tensor;

namespace gpu {

export struct Kernel {
    std::string code {};
    WGPUCommandBuffer commandBuffer {};
};

export struct Context {
    WGPUInstance instance {};
    WGPUAdapter adapter {};
    WGPUDevice device {};
    WGPUQueue queue {};

    ~Context()
    {
        if (queue) {
            wgpuQueueRelease(queue);
        }

        if (device) {
            wgpuDeviceRelease(device);
        }

        if (adapter) {
            wgpuAdapterRelease(adapter);
        }

        if (instance) {
            wgpuInstanceRelease(instance);
        }
    }

    Tensor CreateTensor(Shape shape)
    {
        auto bufferDesc = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
            .size = sizeof(float) * shape.size(),
        };

        auto buffer = wgpuDeviceCreateBuffer(device, &bufferDesc);

        auto tensor = Tensor {
            .buffer = buffer,
            .shape = std::move(shape),
        };

        return tensor;
    }

    Tensor CreateTensor(Shape shape, const float* data)
    {
        auto tensor = CreateTensor(std::move(shape));

        // wgpuQueueWriteBuffer(queue, tensor.buffer, 0, data, tensor.size_in_bytes());

        return tensor;
    }

    Kernel CreateKernel(std::string shaderScript, std::span<Tensor> tensors, size_t workgroupSize)
    {
        auto kernel = Kernel {
            .code = std::move(shaderScript),
        };

        // Create layout entries for buffers.
        auto layoutEntries = std::vector<WGPUBindGroupLayoutEntry>(tensors.size());
        for (auto i = 0u; i < tensors.size(); ++i) {
            layoutEntries[i] = WGPUBindGroupLayoutEntry {
                .binding = i,
                .visibility = WGPUShaderStage_Compute,
                .buffer = WGPUBufferBindingLayout {
                    .type = WGPUBufferBindingType_Storage,
                    .minBindingSize = tensors[i].size_in_bytes(),
                },
            };
        }

        auto layoutDesc = WGPUBindGroupLayoutDescriptor {
            .entryCount = layoutEntries.size(),
            .entries = layoutEntries.data(),
        };

        auto layout = wgpuDeviceCreateBindGroupLayout(device, &layoutDesc);

        // Create bind group entries.
        auto bindGroupEntries = std::vector<WGPUBindGroupEntry>(tensors.size());
        for (auto i = 0u; i < tensors.size(); ++i) {
            bindGroupEntries[i] = WGPUBindGroupEntry {
                .binding = i,
                .buffer = tensors[i].buffer,
                .size = tensors[i].size_in_bytes(),
            };
        }

        auto bindGroupDesc = WGPUBindGroupDescriptor {
            .layout = layout,
            .entryCount = bindGroupEntries.size(),
            .entries = bindGroupEntries.data(),
        };

        auto bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);

        // Create pipeline.
        auto pipelineLayoutDesc = WGPUPipelineLayoutDescriptor {
            .bindGroupLayoutCount = 1,
            .bindGroupLayouts = &layout,
        };

        auto pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDesc);

        // Create wgsl
        auto wgslDesc = WGPUShaderModuleWGSLDescriptor {
            .chain = WGPUChainedStruct {
                .sType = WGPUSType_ShaderModuleWGSLDescriptor,
            },
            .code = kernel.code.c_str(),
        };

        auto shaderModuleDesc = WGPUShaderModuleDescriptor {
            .nextInChain = &wgslDesc.chain,
            .label = "shader",
        };

        auto computePipelineDesc = WGPUComputePipelineDescriptor {
            .layout = pipelineLayout,
            .compute = WGPUProgrammableStageDescriptor {
                .module = wgpuDeviceCreateShaderModule(device, &shaderModuleDesc),
                .entryPoint = "main",
            },
        };

        auto computePipeline = wgpuDeviceCreateComputePipeline(device, &computePipelineDesc);

        // reset command buffer.
        auto commandEncoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
        auto computePassEncoder = wgpuCommandEncoderBeginComputePass(commandEncoder, nullptr);
        wgpuComputePassEncoderSetPipeline(computePassEncoder, computePipeline);
        wgpuComputePassEncoderSetBindGroup(computePassEncoder, 0, bindGroup, 0, nullptr);
        // wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder, workgroupSize, 1, 1);
        wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder, (3072 + (workgroupSize - 1)) / workgroupSize, 1, 1);
        wgpuComputePassEncoderEnd(computePassEncoder);

        kernel.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);

        std::array<float, 3072> inputArr;
        for (size_t i = 0; i < 3072; i++) {
            // Populate input array with a range of dummy values
            inputArr[i] = static_cast<float>(i);
        }
        wgpuQueueWriteBuffer(queue, tensors[0].buffer, 0, inputArr.data(), inputArr.size() * sizeof(float));

        struct CompilationInfo {
            bool ended {};
        } compilationInfo {};

        auto cb = [](WGPUCompilationInfoRequestStatus status, WGPUCompilationInfo const* compilationInfo, void* userData) {
            if (compilationInfo) {
                auto pCompilationInfo = reinterpret_cast<CompilationInfo*>(userData);
                for (uint32_t i = 0; i < compilationInfo->messageCount; ++i) {
                    printf("Message %d: %s\n", i, compilationInfo->messages[i].message);
                }
                pCompilationInfo->ended = true;
            }
        };

        wgpuShaderModuleGetCompilationInfo(computePipelineDesc.compute.module, cb, &compilationInfo);

        while (!compilationInfo.ended) {
            wgpuInstanceProcessEvents(instance);
        }

        return kernel;
    }

    void DispatchKernel(const Kernel& kernel)
    {
        auto promise = std::promise<void> {};
        auto future = promise.get_future();

        // Submit the command buffer.
        wgpuQueueSubmit(queue, 1, &kernel.commandBuffer);
        wgpuQueueOnSubmittedWorkDone(queue, [](WGPUQueueWorkDoneStatus status, void* data) {
            auto *pPromise = reinterpret_cast<std::promise<void>*>(data);
            pPromise->set_value(); }, &promise);

        while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            wgpuInstanceProcessEvents(instance);
        }
    }

    std::vector<float> ToCpu(Tensor& tensor)
    {
        // Create readback buffer.
        auto readbackBufferDescriptor = WGPUBufferDescriptor {
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
            .size = tensor.size_in_bytes(),
        };
        auto readbackBuffer = wgpuDeviceCreateBuffer(device, &readbackBufferDescriptor);

        auto commandEncoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, tensor.buffer, 0, readbackBuffer, 0, tensor.size_in_bytes());

        auto promise = std::promise<void> {};
        auto future = promise.get_future();

        auto commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
        wgpuQueueSubmit(queue, 1, &commandBuffer);
        wgpuQueueOnSubmittedWorkDone(queue, [](WGPUQueueWorkDoneStatus status, void* data) {
            auto *pPromise = reinterpret_cast<std::promise<void>*>(data);
            pPromise->set_value(); }, &promise);

        while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            wgpuInstanceProcessEvents(instance);
        }

        promise = std::promise<void> {};
        future = promise.get_future();

        wgpuBufferMapAsync(readbackBuffer, WGPUMapMode_Read, 0, tensor.size_in_bytes(), [](WGPUBufferMapAsyncStatus status, void* captureData) {
            auto *pPromise = reinterpret_cast<std::promise<void>*>(captureData);
            pPromise->set_value(); }, &promise);

        while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            wgpuInstanceProcessEvents(instance);
        }
    
        auto* pMemData = (float*)(wgpuBufferGetConstMappedRange(readbackBuffer, 0, tensor.size_in_bytes()));
        auto output = std::vector<float> { pMemData, pMemData + tensor.shape.size() };
        return output;
    }
};

export Context CreateContext()
{
    Context context {};

    // Create instance.
    context.instance = wgpuCreateInstance(nullptr);

    // Request adapter.
    struct AdapterData {
        WGPUAdapter adapter {};
        volatile bool ended {};
    } adapterData {};

    auto onRequestAdapter = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, char const* message, void* pUserData) {
        auto pData = reinterpret_cast<AdapterData*>(pUserData);
        pData->adapter = adapter;
        pData->ended = true;
    };

    wgpuInstanceRequestAdapter(context.instance, nullptr, onRequestAdapter, &adapterData);

    while (!adapterData.ended) {
        wgpuInstanceProcessEvents(context.instance);
    }

    context.adapter = adapterData.adapter;

    // Request device
    struct DeviceData {
        WGPUDevice device {};
        volatile bool ended {};
    } deviceData {};

    auto onRequestDevice = [](WGPURequestDeviceStatus status, WGPUDevice device, char const* message, void* pUserData) {
        auto pData = reinterpret_cast<DeviceData*>(pUserData);
        pData->device = device;
        pData->ended = true;
    };

    wgpuAdapterRequestDevice(context.adapter, nullptr, onRequestDevice, &deviceData);

    while (!deviceData.ended) {
        wgpuInstanceProcessEvents(context.instance);
    }

    context.device = deviceData.device;

    // Get device queue
    context.queue = wgpuDeviceGetQueue(context.device);

    // All done.
    return context;
}

}
