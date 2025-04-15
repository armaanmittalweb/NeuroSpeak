
# NeuroSpeak Model Optimization Guide

This guide covers techniques to optimize NeuroSpeak models for improved inference speed and reduced memory footprint without sacrificing accuracy.

## Overview

NeuroSpeak's optimization pipeline achieves up to 40% reduction in inference time through several techniques:

1. **Quantization**: Converting model weights from 32-bit float to 8-bit integer
2. **Pruning**: Removing unnecessary connections in the neural network
3. **ONNX Conversion**: Converting to ONNX format for optimized runtime
4. **Parallel Processing**: Utilizing multi-core capabilities for faster inference

## Quick Start

Optimize a trained model with default settings:

```bash
python -m neurospeak.optimize --model-path ./trained_model --data-dir ./data --output-dir ./optimized_model
```

## Optimization Techniques

### Quantization

Quantization reduces precision of weights and activations, significantly reducing model size and speeding up inference:

```bash
python -m neurospeak.optimize --model-path ./trained_model --data-dir ./data --output-dir ./optimized_model
```

By default, this applies INT8 quantization. For advanced options:

```bash
python -m neurospeak.optimize --model-path ./trained_model --config configs/quantization.yaml
```

Quantization typically reduces model size by 75% with minimal impact on accuracy.

### Pruning

Pruning removes redundant connections in the neural network. This requires retraining:

```bash
python -m neurospeak.optimize --model-path ./trained_model --data-dir ./data --output-dir ./optimized_model --prune
```

Pruning parameters can be configured in the optimization config file. Typical sparsity levels range from 30% to 70%.

### ONNX Runtime Conversion

Converting to ONNX format enables platform-specific optimizations:

```bash
python -m neurospeak.optimize --model-path ./trained_model --data-dir ./data --output-dir ./optimized_model --to-onnx
```

ONNX Runtime provides significant performance improvements especially on CPU deployments.

### Parallel Processing

For multi-core environments, enable parallel processing:

```bash
python -m neurospeak.optimize --model-path ./trained_model --data-dir ./data --output-dir ./optimized_model --parallel --num-workers 4
```

This distributes inference across multiple CPU cores for batch processing.

## Combined Optimization

To apply all optimization techniques together:

```bash
python -m neurospeak.optimize --model-path ./trained_model --data-dir ./data --output-dir ./optimized_model --prune --num-workers 4
```

## Benchmarking

After optimization, benchmark your model:

```bash
python -m neurospeak.benchmark --model-path ./optimized_model/optimized_model --data-dir ./data/benchmark
```

The benchmark script measures:
- Inference time per sample
- Memory consumption
- Throughput (samples per second)
- Accuracy metrics compared to original model

## Deployment Considerations

### Mobile Deployment

For mobile deployment, focus on:
- Quantization (essential)
- Pruning (30-50% sparsity)
- TensorFlow Lite conversion

### Server Deployment

For server deployment, prioritize:
- ONNX Runtime conversion
- Parallel processing
- GPU acceleration where available

### Edge Devices

For edge devices:
- Quantization (INT8)
- Heavy pruning (up to 70%)
- Consider model distillation

## Troubleshooting

### Accuracy Loss After Optimization

- Use quantization-aware training
- Reduce pruning sparsity
- Fine-tune after pruning
- Use representative calibration data

### Slow Inference Despite Optimization

- Check CPU/GPU utilization
- Optimize batch size
- Evaluate data preprocessing bottlenecks
- Consider hardware acceleration options
## Case Study: Production Optimization

Our production deployment achieved 40% reduction in inference time using the following approach:

1. **Quantization**: Applied INT8 quantization after quantization-aware fine-tuning
2. **Parallel Processing**: Distributed inference across 8 CPU cores
3. **Operator Fusion**: Optimized computational graph by fusing operations
4. **Cache Management**: Implemented efficient memory access patterns

This combination maintained accuracy within 1% of the original model while significantly improving throughput.

## Custom Optimization Strategies

### Model Distillation

Knowledge distillation transfers knowledge from a larger "teacher" model to a smaller "student" model:

```bash
python -m neurospeak.distillation --teacher-model ./trained_model --student-config configs/small_model.yaml --data-dir ./data
```

### Dynamic Tensor Rematerialization

For memory-constrained environments:

```bash
python -m neurospeak.optimize --model-path ./trained_model --data-dir ./data --enable-rematerialization --mem-limit 4GB
```

### Custom Hardware Optimizations

For specialized hardware (TPUs, DSPs, etc.), see our hardware-specific guides in the `/docs/hardware/` directory.

## Measuring Impact

Always measure the impact of optimizations on:

1. Inference time (ms/sample)
2. Model size (MB)
3. Accuracy metrics (MCD, WER)
4. Power consumption (for mobile/edge)

## Conclusion

Proper optimization is essential for real-time EEG-to-speech conversion. Our optimization pipeline enables NeuroSpeak to run efficiently on a variety of hardware platforms while maintaining high accuracy.