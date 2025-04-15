# NeuroSpeak Training Guide

This document provides detailed instructions for training NeuroSpeak models.

## Prerequisites

Before training, ensure you have:

- Python 3.8+
- TensorFlow 2.7+
- Required Python packages installed (see `requirements.txt`)
- Sufficient GPU resources (recommended: NVIDIA GPU with 16GB+ VRAM)
- Prepared EEG-speech paired dataset (see [data documentation](data.md))

## Quick Start

The simplest way to train a model with default parameters:

```bash
python -m neurospeak.train --data-dir /path/to/dataset --output-dir ./trained_model
```

## Training with Configuration Files

For more control, use a configuration YAML file:

```bash
python -m neurospeak.train --config configs/advanced_training.yaml
```

## Configuration Options

### Basic Options

- `data_dir`: Directory containing training data
- `output_dir`: Directory to save trained model
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Initial learning rate

### Advanced Options

See `configs/advanced_training.yaml` for all available options, including:

- Model architecture settings (RNN units, Transformer layers, etc.)
- Optimization parameters
- Learning rate schedules
- Data augmentation settings
- Regularization techniques

## Distributed Training

For large-scale training across multiple GPUs:

```bash
python -m neurospeak.distributed_train --config configs/distributed_training.yaml --num-gpus 4
```

## Transfer Learning

To fine-tune an existing model:

```bash
python -m neurospeak.train --config configs/fine_tuning.yaml --pretrained-model ./models/base_model
```

## Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir ./trained_model/logs
```

Key metrics to monitor:
- Loss curves (training and validation)
- Mel-cepstral distortion
- Learning rate changes
- Model parameter statistics

## Early Stopping and Checkpoints

The training script includes:
- Early stopping based on validation loss
- Regular model checkpoints
- Learning rate reduction on plateau

## Hyperparameter Tuning

For hyperparameter optimization:

```bash
python -m neurospeak.hyperparameter_tuning --config configs/tuning.yaml
```

## Common Issues and Solutions

### Out of Memory Errors

- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Reduce model size

### Slow Training

- Enable mixed precision
- Optimize dataset pipeline
- Use data caching
- Update TensorFlow/CUDA versions

### Overfitting

- Increase dropout rate
- Add data augmentation
- Apply weight decay
- Reduce model capacity

## Example Training Scenarios

### Basic Training

```bash
python -m neurospeak.train --data-dir ./data --output-dir ./models/basic --epochs 50 --batch-size 32
```

### Advanced Training with Augmentation

```bash
python -m neurospeak.train --config configs/advanced_training.yaml
```

### Fine-tuning for a Specific Subject

```bash
python -m neurospeak.train --config configs/fine_tuning.yaml --data-dir ./data/subject_007 --pretrained-model ./models/base_model
```

## Performance Metrics

After training, evaluate your model using:

```bash
python -m neurospeak.evaluate --model-path ./trained_model --test-data ./data/test
```

## Next Steps

After successful training:
- [Optimize your model](optimization.md) for deployment
- [Run inference](inference.md) on new EEG data
- [Evaluate performance](evaluation.md) with standard metrics


