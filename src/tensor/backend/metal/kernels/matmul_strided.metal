#include <metal_stdlib>
using namespace metal;

kernel void matmul_strided(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& lhs_row_stride [[buffer(6)]],
    constant uint& lhs_col_stride [[buffer(7)]],
    constant uint& rhs_row_stride [[buffer(8)]],
    constant uint& rhs_col_stride [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        uint lhs_idx = row * lhs_row_stride + k * lhs_col_stride;
        uint rhs_idx = k * rhs_row_stride + col * rhs_col_stride;
        sum += lhs[lhs_idx] * rhs[rhs_idx];
    }

    output[row * N + col] = sum;
}
