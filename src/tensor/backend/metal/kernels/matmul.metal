#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += lhs[row * K + k] * rhs[k * N + col];
    }

    output[row * N + col] = sum;
}
