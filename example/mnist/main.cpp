#include <coroutine>
#include <vector>

import cnn;
import network;

cnn::Promise<void> co_main(cnn::Instance instance)
{
    const size_t kInputNodes = 784;
    const size_t kHiddenNodes = 200;
    const size_t kOutputNodes = 10;
    const float kLearningRate = 0.1f;

    auto network = Network { std::move(instance), kInputNodes, kHiddenNodes, kOutputNodes, kLearningRate };
    co_return;
}

int main()
{
    cnn::Main(co_main);
    return 0;
}