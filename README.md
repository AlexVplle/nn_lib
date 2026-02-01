# nn_lib

A high-performance neural network library written in Rust with support for both CPU and Metal GPU backends.

## Features

- **Multiple Backends**: CPU (via ndarray) and Metal GPU acceleration for Apple Silicon
- **Layer Types**:
  - Dense (fully connected) - ✅ Tensor-based
  - Activation layers - ✅ Tensor-based
  - Convolutional, Pooling - ⚠️ Legacy implementation (Tensor migration in progress)
- **Activations**: ReLU, Softmax, Sigmoid, Tanh
- **Optimizers**: Gradient Descent (SGD)
- **Loss Functions**: Cross Entropy, Mean Squared Error
- **Metrics**: Accuracy, Precision, Recall, F1 Score, and more

## Tensor and Metal Implementation

The Tensor implementation and Metal backend are largely inspired by [Candle](https://github.com/huggingface/candle).

**Additional GPU backends (CUDA, WebGPU) coming soon.**

## Performance

Benchmarked on MNIST (784→256→128→10 MLP, 10 epochs, batch size 128):

| Backend | Time (10 epochs) | Speed | Test Accuracy |
|---------|------------------|-------|---------------|
| **Metal (M4 Pro)** | 18.57s | 1.86s/epoch | 93.61% |
| **CPU** | ~211s | ~21s/epoch | - |

## Examples

### MNIST MLP Training

```bash
# CPU backend
RUST_LOG=info cargo run --example mlp_mnist --release

# Metal backend
RUST_LOG=info cargo run --example mlp_mnist --release --features metal
```

The example automatically loads and processes the MNIST dataset, trains a simple MLP, and reports training progress and test accuracy.

## Current Status

### ✅ Implemented (Tensor-based)
- Dense (fully connected) layers
- Activation layers (ReLU, Softmax, Sigmoid, Tanh)
- Gradient Descent optimizer
- Cross Entropy and MSE loss functions
- Comprehensive metrics system
- Metal and CPU backends

### ⚠️ Migration in Progress
- Convolutional layers (legacy ndarray implementation)
- Pooling layers (legacy ndarray implementation)
- Reshape layers (legacy ndarray implementation)

These layers currently use the old ndarray-based implementation and will be migrated to the new Tensor system soon.

## Roadmap

### Coming Soon
- **Autograd**: Automatic differentiation
- **Conv/Pooling Tensor Migration**: Update convolutional and pooling layers to use Tensor backend
- **CUDA Backend**: NVIDIA GPU support
- **WebGPU Backend**: Browser and cross-platform support

### Planned
- Additional optimizers (Adam, RMSprop, AdamW)
- Dropout and other regularization techniques
- More layer types (LSTM, GRU, Transformer, Attention)
- Model serialization/deserialization
- ONNX export

## Current Limitations

- Manual gradient computation (autograd coming soon)
- Convolutional and pooling layers not yet migrated to Tensor backend
- Limited to sequential models

## Authors

- Adrien Pelfresne
- Alexis Vapaille
