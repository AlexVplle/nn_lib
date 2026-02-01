#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define TILE_SIZE 32

kernel void matmul(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    threadgroup float lhs_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float rhs_tile[TILE_SIZE][TILE_SIZE];

    uint row = gid.y * TILE_SIZE + tid.y;
    uint col = gid.x * TILE_SIZE + tid.x;

    float sum = 0.0;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint lhs_col = t * TILE_SIZE + tid.x;
        uint rhs_row = t * TILE_SIZE + tid.y;

        if (row < M && lhs_col < K) {
            lhs_tile[tid.y][tid.x] = lhs[row * K + lhs_col];
        } else {
            lhs_tile[tid.y][tid.x] = 0.0;
        }

        if (rhs_row < K && col < N) {
            rhs_tile[tid.y][tid.x] = rhs[rhs_row * N + col];
        } else {
            rhs_tile[tid.y][tid.x] = 0.0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += lhs_tile[tid.y][k] * rhs_tile[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}
