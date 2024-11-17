module;

#include <coroutine>
#include <functional>
#include <memory>
#include <webgpu/webgpu.h>

export module gpu_matrix:gpu_instance;
import :adapter;
import :ref_ptr;
import :promise;
import :log;

namespace gpu_matrix {

export class GpuInstance {
public:
    GpuInstance() = default;

    GpuInstance(WGPUInstance instance)
        : m_pInstance { instance }
    {
    }

    Promise<Adapter> RequestAdapter()
    {
        using AdapterPtr = ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease>;
        using DevicePtr = ref_ptr<WGPUDevice, wgpuDeviceAddRef, wgpuDeviceRelease>;

        // Request adapter.
        auto adapterPromise = Promise<AdapterPtr>();
        wgpuInstanceRequestAdapter(m_pInstance.get(), nullptr, [](WGPURequestAdapterStatus status, WGPUAdapter adapter, char const* message, void* pUserData) { (*Promise<AdapterPtr>::GetState(pUserData))->SetValue(AdapterPtr { adapter }); }, adapterPromise.GetState().release());
        auto pAdapter = co_await adapterPromise;

        // Request device.
        auto devicePromise = Promise<DevicePtr>();
        wgpuAdapterRequestDevice(pAdapter.get(), nullptr, [](WGPURequestDeviceStatus status, WGPUDevice device, char const* message, void* pUserData) { (*Promise<DevicePtr>::GetState(pUserData))->SetValue(DevicePtr { device }); }, devicePromise.GetState().release());
        auto pDevice = co_await devicePromise;

        // All done.
        co_return { pAdapter.release(), pDevice.release() };
    }

    void ProcessEvents()
    {
        wgpuInstanceProcessEvents(m_pInstance.get());
    }

private:
    ref_ptr<WGPUInstance, wgpuInstanceAddRef, wgpuInstanceRelease> m_pInstance {};
};

export void Main(std::function<Promise<void>(GpuInstance)> func)
{
    auto instance = GpuInstance { wgpuCreateInstance(nullptr) };
    auto promise = func(instance);
    while (!promise.await_ready()) {
        instance.ProcessEvents();
    }
}

}