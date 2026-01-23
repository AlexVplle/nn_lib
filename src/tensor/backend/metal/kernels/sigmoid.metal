#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = 1.0f / (1.0f + exp(-input[id]));
}
