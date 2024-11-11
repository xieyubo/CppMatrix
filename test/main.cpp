#include <array>
#include <cassert>
#include <coroutine>
#include <cstdio>
#include <cstring>
#include <webgpu/webgpu.h>

/*
 * Approximate GELU kernel definition, implemented as a WGSL.
 * In general GPU device code for WEBGPU is written in the WGSL domain specific
 * language.
 *
 * Here inp and out correspond to bindings 0 and 1 respectively. In the main
 * code, we create buffers for these bindings and populate them with data.
 *
 */
const char* kShaderGELU = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    // Ensure we do not access out of bounds
    if (i < 3072) {
        let x: f32 = inp[i];
        let cube: f32 = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR * (x + cube)));
    }
}
)";

#include <array>
#include <span>
#include <vector>

import gpu;

inline size_t cdiv(size_t n, size_t d) { return (n + d - 1) / d; }

gpu::Promise<void> co_main(gpu::Instance instance)
{
    constexpr size_t N = 3072;
    std::array<float, N> inputArr, outputArr;
    for (int i = 0; i < N; ++i) {
        inputArr[i] = i / 10.0;
    }
    for (auto i = 0u; i < 10; ++i) {
        printf("input[%d]: %f\n", i, inputArr[i]);
    }
    printf("...\n");

    auto adapter = co_await instance.RequestAdapter();
    auto input = adapter.CreateBuffer(gpu::Dimension { N });
    auto output = adapter.CreateBuffer(gpu::Dimension { N });

    {
        std::array<float, N> inputArr, outputArr;
        for (int i = 0; i < N; ++i) {
            inputArr[i] = i / 10.0;
        }
        for (auto i = 0u; i < 10; ++i) {
            printf("input[%d]: %f\n", i, inputArr[i]);
        }
        printf("...\n");

        auto context = gpu::CreateContext();
        auto input = context.CreateTensor(gpu::Shape { N }, inputArr.data());
        auto output = context.CreateTensor(gpu::Shape { N });

        auto k = std::vector<gpu::Tensor> { input, output };
        auto kernel = context.CreateKernel(kShaderGELU, { k.begin(), k.end() }, cdiv(N, 256));
        context.DispatchKernel(kernel);

        // copy output to cpu
        auto res = context.ToCpu(output);
        for (auto i = 0u; i < 10; ++i) {
            printf("output[%d]: %f\n", i, res[i]);
        }
        printf("...\n");
    }
    co_return;
}

int main()
{
    gpu::Main(co_main);
    return 0;
}
