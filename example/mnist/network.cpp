module;

#include <coroutine>
#include <cstdlib>
#include <ctime>
#include <span>
#include <stdexcept>
#include <vector>

import cpp_matrix;

using namespace cpp_matrix;

export module network;

export class Network {
public:
    Network(GpuInstance gpuInstance, size_t inputNodes, size_t hiddenNodes, size_t outputNodes, float learningRate)
        : m_gpuInstance { std::move(gpuInstance) }
        , m_inputNodes { inputNodes }
        , m_hiddenNodes { hiddenNodes }
        , m_outputNodes { outputNodes }
        , m_learningRate { learningRate }
    {
    }

    Promise<void> Train(std::vector<float> input, std::vector<float> target)
    {
        co_await InitializeGpu();

        if (input.size() != m_inputNodes) {
            throw std::runtime_error { "input size is incorrect." };
        }

        /*if (target.size() != m_outputNodes) {
            throw std::runtime_error { "output size is incorrect." };
        }*/

        auto inputTensor = Matrix { m_inputNodes, 1 };
        inputTensor.Write(std::span<float> { input.begin(), input.end() });

        auto res = m_weightIH * inputTensor;
        auto data = res.Read();
        co_return;
    }

private:
    Promise<void> InitializeGpu()
    {
        if (!m_adapter) {
            m_adapter = m_gpuInstance.GetAdapter();
        }

        if (!m_weightIH) {
            m_weightIH = GenerateWeights(m_hiddenNodes, m_inputNodes);
        }

        if (!m_weightHO) {
            m_weightHO = GenerateWeights(m_outputNodes, m_hiddenNodes);
        }

        co_return;
    }

    Matrix GenerateWeights(size_t row, size_t column)
    {
        std::srand(std::time(0));

        // Generate random data.
        std::vector<float> data {};
        data.resize(row * column);
        for (auto i = 0u; i < data.size(); ++i) {
            data[i] = i + 1; //(float)((double)std::rand() / RAND_MAX);
        }

        // Create tensor.
        auto weights = Matrix { row, column };
        weights.Write(std::span { data.begin(), data.end() });
        return weights;
    }

    GpuInstance m_gpuInstance {};
    size_t m_inputNodes {};
    size_t m_hiddenNodes {};
    size_t m_outputNodes {};
    float m_learningRate {};
    Adapter m_adapter {};
    Matrix m_weightIH {};
    Matrix m_weightHO {};
};
