module;

#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <vector>

import cpp_matrix;

using namespace cpp_matrix;

export module neural_network;

template <typename T>
static bool AreEqual(T f1, T f2)
{
    return (std::fabs(f1 - f2) <= std::numeric_limits<T>::epsilon() * std::fmax(std::fabs(f1), std::fabs(f2)));
}

export class NeuralNetwork {
public:
    NeuralNetwork(size_t inputNodes, size_t hiddenNodes, size_t outputNodes, float learningRate)
        : m_inodes { inputNodes }
        , m_hnodes { hiddenNodes }
        , m_onodes { outputNodes }
        , m_wih { Matrix::Random(hiddenNodes, inputNodes) - 0.5f }
        , m_who { Matrix::Random(outputNodes, hiddenNodes) - 0.5f }
        , m_lr { learningRate }
    {
    }

    void Train(std::vector<float> inputs_list, std::vector<float> targets_list)
    {
        // convert inputs list to matrix
        auto inputs = Matrix { m_inodes, /*column=*/1, inputs_list };
        auto targets = Matrix { m_onodes, /*column=*/1, targets_list };

        // calculate signals into hidden layer
        auto hidden_inputs = m_wih * inputs;

        // calculate the signals emerging from hidden layer
        auto hidden_outputs = hidden_inputs.Sigmoid();

        //  calculate signals into final output layer
        auto final_inputs = m_who * hidden_outputs;

        // calculate the signals emerging from final output layer
        auto final_outputs = final_inputs.Sigmoid();

        // output layer error is the (target - actual)
        auto output_errors = targets - final_outputs;

        // hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        auto hidden_errors = m_who.Transpose() * output_errors;

        // update the weights for the links between the hidden and output layers
        m_who += m_lr * (output_errors.ElementProduct(final_outputs).ElementProduct(1.0f - final_outputs) * hidden_outputs.Transpose());

        // update the weights for the links between the input and hidden layers
        m_wih += m_lr * ((hidden_errors.ElementProduct(hidden_outputs).ElementProduct(1.0f - hidden_outputs) * inputs.Transpose()));
    }

    std::vector<float> Query(std::vector<float> inputs_list)
    {
        // convert inputs list to matrix
        auto inputs = Matrix { m_inodes, /*column=*/1, inputs_list };

        // caculate signals into hidden layer
        auto hidden_inputs = m_wih * inputs;

        // calculate the signals emerging from hidden layer
        auto hidden_outputs = hidden_inputs.Sigmoid();

        // calculate signals into final output layer
        auto final_inputs = m_who * hidden_outputs;

        // caculate the signals emerging from final output layer
        auto final_outputs = final_inputs.Sigmoid();

        return final_outputs.Read();
    }

private:
    size_t m_inodes {};
    size_t m_hnodes {};
    size_t m_onodes {};
    Matrix m_wih {};
    Matrix m_who {};
    float m_lr {};
};
