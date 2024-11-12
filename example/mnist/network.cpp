module;

#include <coroutine>
#include <vector>

import cnn;

using namespace cnn;

export module network;

export class Network {
public:
    Network(Instance gpuInstance, size_t inputNodes, size_t hiddenNodes, size_t outputNodes, float learningRate)
        : m_gpuInstance { std::move(gpuInstance) }
    {
    }

    Promise<void> Train()
    {
        co_await InitializeGpu();
    }

private:
    Promise<void> InitializeGpu()
    {
        if (!m_adapter) {
            m_adapter = co_await m_gpuInstance.RequestAdapter();
        }
    }

    Instance m_gpuInstance {};
    Adapter m_adapter {};
};
