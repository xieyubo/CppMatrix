module;

#include <cassert>
#include <coroutine>
#include <functional>
#include <memory>
#include <webgpu/webgpu.h>

export module cpp_matrix:gpu_instance;
import :adapter;
import :ref_ptr;
import :promise;
import :log;

namespace cpp_matrix {

export class GpuInstance {
public:
    static GpuInstance& GetInstance()
    {
        static GpuInstance s_gpuInstance { wgpuCreateInstance(nullptr) };
        return s_gpuInstance;
    }

    GpuInstance() = default;

    GpuInstance(WGPUInstance instance)
        : m_pInstance { instance }
    {
    }

    Adapter GetAdapter()
    {
        static Adapter adapter { RequestAdapter().await_resume() };
        return adapter;
    }

    void ProcessEvents()
    {
        wgpuInstanceProcessEvents(m_pInstance.get());
    }

private:
    Promise<Adapter> RequestAdapter()
    {
        using AdapterPtr = ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease>;
        using DevicePtr = ref_ptr<WGPUDevice, wgpuDeviceAddRef, wgpuDeviceRelease>;

        // Request adapter.
        auto adapterPromise = Promise<AdapterPtr>();
        wgpuInstanceRequestAdapter(m_pInstance.get(), nullptr, [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* pUserData) { (*Promise<AdapterPtr>::GetState(pUserData))->SetValue(AdapterPtr { adapter }); }, adapterPromise.GetState().release());
        auto pAdapter = co_await adapterPromise;

        // Request device.
        auto devicePromise = Promise<DevicePtr>();
        WGPURequiredLimits requiredLimis = WGPU_REQUIRED_LIMITS_INIT;
        // Currently we only support offset aligment is 16.
        requiredLimis.limits.minStorageBufferOffsetAlignment = 16;
        WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
        descriptor.requiredLimits = &requiredLimis;
        wgpuAdapterRequestDevice(pAdapter.get(), &descriptor, [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* pUserData) { 
            assert(status == WGPURequestDeviceStatus_Success);
            (*Promise<DevicePtr>::GetState(pUserData))->SetValue(DevicePtr { device }); }, devicePromise.GetState().release());
        auto pDevice = co_await devicePromise;

        // All done.
        co_return { pAdapter.release(), pDevice.release() };
    }

    ref_ptr<WGPUInstance, wgpuInstanceAddRef, wgpuInstanceRelease> m_pInstance {};
};

void ProcessGpuInstanceEvents()
{
    GpuInstance::GetInstance().ProcessEvents();
}

}