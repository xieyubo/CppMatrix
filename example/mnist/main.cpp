#include <coroutine>
#include <vector>

import cpp_matrix;
import network;

cpp_matrix::Promise<void> co_main(cpp_matrix::GpuInstance instance)
{
    const size_t kInputNodes = 3;//784;
    const size_t kHiddenNodes = 4; //200;
    const size_t kOutputNodes = 3; //10;
    const float kLearningRate = 0.1f;

    auto network = Network { std::move(instance), kInputNodes, kHiddenNodes, kOutputNodes, kLearningRate };

    std::vector<float> input {1, 2, 3};
    co_await network.Train(std::move(input), {});
}

int main()
{
    co_main(cpp_matrix::GpuInstance().GetInstance()).await_resume();
    return 0;
}