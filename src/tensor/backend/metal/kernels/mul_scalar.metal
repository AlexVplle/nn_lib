#include <metal_stdlib>
using namespace metal;

kernel void mul_scalar(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input[id] * scalar;
}
