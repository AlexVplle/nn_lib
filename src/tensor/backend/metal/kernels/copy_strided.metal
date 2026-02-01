#include <metal_stdlib>
using namespace metal;

kernel void copy_strided(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    constant uint* strides [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& total_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_size) return;

    uint indices[8];
    uint remaining = gid;

    for (int d = ndim - 1; d >= 0; d--) {
        indices[d] = remaining % shape[d];
        remaining /= shape[d];
    }

    uint input_offset = 0;
    for (uint d = 0; d < ndim; d++) {
        input_offset += indices[d] * strides[d];
    }

    output[gid] = input[input_offset];
}
