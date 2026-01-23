#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& vector_size [[buffer(2)]],
    uint batch_id [[thread_position_in_grid]]
) {
    uint start = batch_id * vector_size;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (uint i = 0; i < vector_size; i++) {
        max_val = max(max_val, input[start + i]);
    }

    // Compute exp(x - max) and sum
    float sum_exps = 0.0f;
    for (uint i = 0; i < vector_size; i++) {
        float exp_val = exp(input[start + i] - max_val);
        output[start + i] = exp_val;
        sum_exps += exp_val;
    }

    // Normalize
    sum_exps += 1e-10f; // Avoid division by zero
    for (uint i = 0; i < vector_size; i++) {
        output[start + i] /= sum_exps;
    }
}
