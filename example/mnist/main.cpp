#include <cstring>
#include <ctime>
#include <format>
#include <fstream>
#include <span>
#include <sstream>
#include <vector>

import neural_network;
import cpp_matrix;

using namespace cpp_matrix;

struct Options {
    int epochs { 1 };
    std::string training_file;
    std::string test_file;
    bool useF16 {};
    bool useCpuMatrix {};
    bool useGpuMatrix {};
};

static Options parse_options(int argc, char* argv[])
{
    auto options = Options {};
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], "--use-cpu")) {
            options.useCpuMatrix = true;
        } else if (!strcmp(argv[i], "--use-gpu")) {
            options.useGpuMatrix = true;
        } else if (!strcmp(argv[i], "--use-f16")) {
            options.useF16 = true;
        } else if (!strcmp(argv[i], "--epochs")) {
            options.epochs = atoi(argv[++i]);
        } else if (options.training_file.empty()) {
            options.training_file = argv[i];
        } else if (options.test_file.empty()) {
            options.test_file = argv[i];
        } else {
            throw std::runtime_error { std::format("Unknown options: {}", argv[i]) };
        }
    }
    return options;
}

template <MatrixElementType T>
std::vector<std::pair<int, std::vector<T>>> read_data_from_file(std::string filename)
{
    std::vector<std::pair<int, std::vector<T>>> datas;
    std::ifstream in { filename };
    std::string line;
    while (getline(in, line)) {
        std::stringstream ss { line };
        std::vector<T> inputs;
        std::string str;
        if (!getline(ss, str, ',')) {
            throw std::runtime_error { "Unexpected input file." };
        }
        auto v = atoi(str.c_str());
        while (getline(ss, str, ',')) {
            inputs.push_back(atof(str.c_str()) / 255.f * 0.99f + 0.01f);
        }
        if (inputs.size() != 784) {
            throw std::runtime_error { "Unexpected input file." };
        }
        datas.emplace_back(v, std::move(inputs));
    }
    return datas;
}

static void print_help(const char* appname)
{
    printf("%s [--use-cpu|--use-gpu] [--use-f16] [--epochs x] training_file test_file\n", appname);
}

template <MatrixElementType T>
void run(NeuralNetwork<T> network, const Options& options)
{
    auto training_data = read_data_from_file<T>(options.training_file);

    for (int i = 0; i < options.epochs; ++i) {
        for (const auto& [v, inputs] : training_data) {
            std::vector<T> targets(10, 0.01f);
            targets[v] = 0.99f;
            network.Train(inputs, targets);
        }
    }

    // test the network
    auto test_data = read_data_from_file<T>(options.test_file);
    int total {}, correct {};
    for (const auto& [v, inputs] : test_data) {
        auto res = network.Query(inputs);
        if (res.size() != 10) {
            throw std::runtime_error { "Bad prediction result." };
        }

        auto maxIndex = 0;
        for (int i = 1; i < res.size(); ++i) {
            if (res[i] > res[maxIndex]) {
                maxIndex = i;
            }
        }
        printf("prediction result: %d, actual result: %d %c\n", maxIndex, v, (maxIndex == v ? 'o' : 'x'));

        ++total;
        if (maxIndex == v) {
            ++correct;
        }
    }
    printf("performance = %g\n", (double)((T)correct / total));
}

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        print_help(argv[0]);
        return 1;
    }

    auto options = parse_options(argc - 1, argv + 1);
    if (options.useGpuMatrix && options.useCpuMatrix) {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::Auto);
    } else if (options.useCpuMatrix) {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::CpuMatrix);
    } else if (options.useGpuMatrix) {
        cpp_matrix::SetDefaultMatrixType(cpp_matrix::MatrixType::WebGpuMatrix);
    }

    const size_t kInputNodes = 784;
    const size_t kHiddenNodes = 200;
    const size_t kOutputNodes = 10;
    const float kLearningRate = 0.1f;

    if (options.useF16) {
        run(NeuralNetwork<std::float16_t> { kInputNodes, kHiddenNodes, kOutputNodes, (std::float16_t)kLearningRate },
            options);
    } else {
        run(NeuralNetwork<std::float32_t> { kInputNodes, kHiddenNodes, kOutputNodes, kLearningRate }, options);
    }

    return 0;
}