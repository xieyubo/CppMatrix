#include <coroutine>
#include <vector>

import cnn;
import network;

cnn::Promise<void> co_main(cnn::Instance instance)
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
    cnn::Main(co_main);
    return 0;
}