#include <coroutine>
#include <gtest/gtest.h>

import cpp_matrix;

using namespace cpp_matrix;

inline size_t cdiv(size_t n, size_t d) { return (n + d - 1) / d; }

TEST(GpuTest, GELU)
{
    std::vector<float> res;

    auto func = [&res](GpuInstance instance) -> cpp_matrix::Promise<void> {
        constexpr const char* kShaderGELU = R"(
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

        constexpr size_t N = 3072;

        std::array<float, N> inputArr, outputArr;
        for (int i = 0; i < N; ++i) {
            inputArr[i] = i;
        }

        auto adapter = instance.GetAdapter();
        auto input = GpuMatrix { 1, N };
        auto output = GpuMatrix { 1, N };

        input.Write(std::span { inputArr });

        auto k = std::vector<GpuMatrix> { input, output };
        co_await adapter.Run(kShaderGELU, { k.begin(), k.end() }, cdiv(N, 256));

        res = co_await output.Read();
        co_return;
    };

    func(cpp_matrix::GpuInstance().GetInstance()).await_resume();
    ASSERT_EQ(res.size(), 3072);
    ASSERT_FLOAT_EQ(res[0], 0.000000);
    ASSERT_FLOAT_EQ(res[1], 0.841192f);
    ASSERT_FLOAT_EQ(res[2], 1.954598f);
    ASSERT_FLOAT_EQ(res[3], 2.996363);
    ASSERT_FLOAT_EQ(res[4], 3.999930);
    ASSERT_FLOAT_EQ(res[5], 5.000000);
    ASSERT_FLOAT_EQ(res[6], 6.000000);
    ASSERT_FLOAT_EQ(res[7], 7.000000);
    ASSERT_FLOAT_EQ(res[8], 8.000000);
    ASSERT_FLOAT_EQ(res[9], 9.000000);
}