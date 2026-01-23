#include <metal_stdlib>
using namespace metal;

kernel void tanh_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = tanh(input[id]);
}
