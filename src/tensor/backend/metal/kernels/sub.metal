#include <metal_stdlib>
using namespace metal;

kernel void sub(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = lhs[id] - rhs[id];
}
