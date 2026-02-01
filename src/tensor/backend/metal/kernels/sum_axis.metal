#include <metal_stdlib>
using namespace metal;

kernel void sum_axis(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  device const uint* input_shape [[buffer(2)]],
  device const uint* output_shape [[buffer(3)]],
  device const uint* input_strides [[buffer(4)]],
  device const uint* output_strides [[buffer(5)]],
  constant uint& axis [[buffer(6)]],
  constant uint& input_ndim [[buffer(7)]],
  constant uint& output_ndim [[buffer(8)]],
  constant uint& axis_size [[buffer(9)]],
  uint out_idx [[thread_position_in_grid]]
) {
  uint out_indices[8];
  uint remaining = out_idx;

  for (uint i = 0; i < output_ndim; i++) {
    out_indices[i] = remaining / output_strides[i];
    remaining %= output_strides[i];
  }

  uint input_indices[8];
  uint out_pos = 0;
  for (uint i = 0; i < input_ndim; i++) {
    if (i == axis) {
      input_indices[i] = 0;
    } else {
      input_indices[i] = out_indices[out_pos];
      out_pos++;
    }
  }

  float sum = 0.0;
  for (uint axis_idx = 0; axis_idx < axis_size; axis_idx++) {
    input_indices[axis] = axis_idx;

    uint in_idx = 0;
    for (uint i = 0; i < input_ndim; i++) {
      in_idx += input_indices[i] * input_strides[i];
    }

    sum += input[in_idx];
  }

  output[out_idx] = sum;
}
