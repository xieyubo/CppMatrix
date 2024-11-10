module;

#include <cstdio>
#include <thread>
#include <webgpu/webgpu.h>

export module gpu:context;

namespace gpu {

export struct Context {
    WGPUInstance instance {};
    WGPUAdapter adapter {};

    ~Context()
    {
        if (adapter) {
            wgpuAdapterRelease(adapter);
        }

        if (instance) {
            wgpuInstanceRelease(instance);
        }
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

    // All done.
    return context;
}

}
