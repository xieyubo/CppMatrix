#include <vector>

import network;

int main()
{
    const size_t kInputNodes = 784;
    const size_t kHiddenNodes = 200;
    const size_t kOutputNodes = 10;
    const float kLearningRate = 0.1f;

    auto network = Network { kInputNodes, kHiddenNodes, kOutputNodes, kLearningRate };
    return 0;
}