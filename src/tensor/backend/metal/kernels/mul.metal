#include <metal_stdlib>
using namespace metal;

kernel void mul(device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* result [[buffer(2)]],
                uint id [[thread_position_in_grid]]) {
    result[id] = a[id] * b[id];
}
