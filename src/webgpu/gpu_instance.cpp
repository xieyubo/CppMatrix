module;

#include <cassert>
#include <functional>
#include <future>
#include <memory>
#include <webgpu/webgpu.h>

export module webgpu:gpu_instance;
import :adapter;
import :gpu_ref_ptr;

namespace webgpu {

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

    GpuAdapter GetAdapter()
    {
        static GpuAdapter adapter { RequestAdapter() };
        return adapter;
    }

    void ProcessEvents()
    {
        wgpuInstanceProcessEvents(m_pInstance.get());
    }

private:
    GpuAdapter RequestAdapter()
    {
        using AdapterPtr = gpu_ref_ptr<WGPUAdapter, wgpuAdapterAddRef, wgpuAdapterRelease>;
        using DevicePtr = gpu_ref_ptr<WGPUDevice, wgpuDeviceAddRef, wgpuDeviceRelease>;

        // Request adapter.
        auto adapterPromise = std::promise<AdapterPtr>();
        auto adapterFuture = adapterPromise.get_future();
        wgpuInstanceRequestAdapter(m_pInstance.get(), nullptr, [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* pUserData) { ((std::promise<AdapterPtr>*)pUserData)->set_value(AdapterPtr { adapter }); }, &adapterPromise);
        auto pAdapter = Wait(adapterFuture);

        // Request device.
        auto devicePromise = std::promise<DevicePtr>();
        auto deviceFuture = devicePromise.get_future();
        WGPURequiredLimits requiredLimis = WGPU_REQUIRED_LIMITS_INIT;
        // Currently we only support offset aligment is 16.
        requiredLimis.limits.minStorageBufferOffsetAlignment = 16;
        WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
        descriptor.requiredLimits = &requiredLimis;
        wgpuAdapterRequestDevice(pAdapter.get(), &descriptor, [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* pUserData) { 
            assert(status == WGPURequestDeviceStatus_Success);
            ((std::promise<DevicePtr>*)pUserData)->set_value(DevicePtr { device }); }, &devicePromise);
        auto pDevice = Wait(deviceFuture);

        // All done.
        return { pAdapter.release(), pDevice.release() };
    }

    template <typename T>
    T Wait(std::future<T>& future)
    {
        while (future.wait_for(std::chrono::milliseconds {}) != std::future_status::ready) {
            ProcessGpuInstanceEvents();
        }
        return future.get();
    }

    gpu_ref_ptr<WGPUInstance, wgpuInstanceAddRef, wgpuInstanceRelease> m_pInstance {};
};

void ProcessGpuInstanceEvents()
{
    GpuInstance::GetInstance().ProcessEvents();
}

}