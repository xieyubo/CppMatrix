module;

#include <functional>
#include <memory>
#include <webgpu/webgpu.h>

export module gpu:instance;
import :adapter;
import :gpu_ref_ptr;
import :promise;
import :log;

namespace gpu {

export class Instance {
public:
    Instance(WGPUInstance instance)
        : m_pInstance { instance }
    {
    }

    Promise<Adapter> RequestAdapter()
    {
        auto promise = Promise<Adapter>();
        wgpuInstanceRequestAdapter(m_pInstance.get(), nullptr, [](WGPURequestAdapterStatus status, WGPUAdapter adapter, char const* message, void* pUserData) {
            auto pState = Promise<Adapter>::GetState(pUserData);
            (*pState)->SetValue(Adapter {adapter}); }, promise.GetState().release());
        return promise;
    }

    void ProcessEvents()
    {
        wgpuInstanceProcessEvents(m_pInstance.get());
    }

private:
    gpu_ref_ptr<WGPUInstance, wgpuInstanceAddRef, wgpuInstanceRelease> m_pInstance {};
};

export void Main(std::function<Promise<void>(Instance)> func)
{
    auto instance = Instance { wgpuCreateInstance(nullptr) };
    auto promise = func(instance);
    while (!promise.await_ready()) {
        instance.ProcessEvents();
    }
}

}