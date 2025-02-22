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

    std::shared_ptr<GpuAdapter> GetAdapter()
    {
        static auto s_adapter { RequestAdapter() };
        return s_adapter;
    }

    void ProcessEvents()
    {
        wgpuInstanceProcessEvents(m_pInstance.get());
    }

private:
    std::shared_ptr<GpuAdapter> RequestAdapter()
    {
        // Request adapter.
        auto adapterPromise = std::promise<GpuAdapterPtr>();
        auto adapterFuture = adapterPromise.get_future();
        wgpuInstanceRequestAdapter(m_pInstance.get(), nullptr,
            { .mode = WGPUCallbackMode_AllowProcessEvents,
                .callback =
                    [](WGPURequestAdapterStatus status, WGPUAdapter adapter, struct WGPUStringView message,
                        void* userdata1, void* userdata2) {
                        ((std::promise<GpuAdapterPtr>*)userdata1)->set_value(GpuAdapterPtr { adapter });
                    },
                .userdata1 = &adapterPromise });
        auto pAdapter = Wait(adapterFuture);

        // Request device.
        try {
            auto pDevice = RequestDevice(pAdapter.get(), { WGPUFeatureName_ShaderF16 });
            return std::make_shared<GpuAdapter>(pAdapter.release(), pDevice.release());
        } catch (std::runtime_error) {
            // F16 might not supported, that's fine, let's create device without F16 feature.
        }

        auto pDevice = RequestDevice(pAdapter.get(), {});
        return std::make_shared<GpuAdapter>(pAdapter.release(), pDevice.release());
    }

    static GpuDevicePtr RequestDevice(WGPUAdapter adapter, std::initializer_list<WGPUFeatureName> features)
    {
        // Request device.
        auto devicePromise = std::promise<GpuDevicePtr>();
        auto deviceFuture = devicePromise.get_future();
        WGPUDeviceDescriptor desc = WGPU_DEVICE_DESCRIPTOR_INIT;
        desc.requiredFeatureCount = features.size();
        desc.requiredFeatures = std::data(features);
        wgpuAdapterRequestDevice(adapter, &desc,
            { .mode = WGPUCallbackMode_AllowProcessEvents,
                .callback =
                    [](WGPURequestDeviceStatus status, WGPUDevice device, struct WGPUStringView message,
                        void* userdata1, void* userdata2) {
                        auto pPromise = (std::promise<GpuDevicePtr>*)userdata1;
                        if (status == WGPURequestDeviceStatus_Success) {
                            pPromise->set_value(GpuDevicePtr { device });
                        } else {
                            pPromise->set_exception(std::make_exception_ptr(
                                std::runtime_error { std::string { message.data, message.data + message.length } }));
                        }
                    },
                .userdata1 = &devicePromise });
        return Wait(deviceFuture);
    }

    template <typename T>
    static T Wait(std::future<T>& future)
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