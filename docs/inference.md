# NeuroSpeak Inference Guide

This guide explains how to run inference with trained NeuroSpeak models to convert EEG signals to speech.

## Quick Start

Basic inference with a pretrained model:

```python
from neurospeak import NeuroSpeakModel, EEGProcessor

# Initialize the EEG processor
eeg_processor = EEGProcessor(sampling_rate=256, channels=64)

# Load pretrained model
model = NeuroSpeakModel.from_pretrained('models/neurospeak_v1.0')

# Process EEG data and generate speech
eeg_data = eeg_processor.load_data('path/to/eeg_recording.edf')
processed_signal = eeg_processor.preprocess(eeg_data)
speech_output = model.generate_speech(processed_signal)

# Save the generated speech
speech_output.save_to_file('output_speech.wav')
```

## Command-Line Interface

Run inference via the command line:

```bash
python -m neurospeak.inference \
    --model-path models/neurospeak_v1.0 \
    --eeg-file path/to/eeg_recording.edf \
    --output-file output_speech.wav \
    --config configs/inference.yaml
```

## Batch Processing

Process multiple EEG files at once:

```bash
python -m neurospeak.batch_inference \
    --model-path models/neurospeak_v1.0 \
    --eeg-dir path/to/eeg_files \
    --output-dir path/to/output \
    --config configs/inference.yaml
```

## Real-time Processing

For near real-time EEG-to-speech conversion:

```python
from neurospeak import NeuroSpeakRealtime

# Initialize real-time processor
rt_processor = NeuroSpeakRealtime(
    model_path='models/neurospeak_v1.0',
    buffer_size=2.0,  # 2 seconds buffer
    step_size=0.5,    # 0.5 seconds step
    device='cpu'      # or 'gpu'
)

# Start processing
rt_processor.connect_to_eeg_stream('LSL_EEG_Stream')
rt_processor.start()

# Later, to stop processing
rt_processor.stop()
```

## Inference Configuration

### Model Selection

You can choose between different model variants:

```python
# Base model
model = NeuroSpeakModel.from_pretrained('models/neurospeak_base')

# Subject-specific model
model = NeuroSpeakModel.from_pretrained('models/neurospeak_subject_007')

# Optimized model for faster inference
model = NeuroSpeakModel.from_pretrained('models/neurospeak_optimized')
```

### Hardware Acceleration

Enable GPU acceleration:

```python
model = NeuroSpeakModel.from_pretrained(
    'models/neurospeak_v1.0',
    device='cuda'  # Use 'cuda' for NVIDIA GPUs, 'cpu' for CPU-only
)
```

### Inference Parameters

Customize the speech generation process:

```python
speech_output = model.generate_speech(
    processed_signal,
    vocoder='waveglow',          # Neural vocoder for high-quality synthesis
    temperature=0.8,             # Control randomness (0.5-1.0)
    length_scale=1.2,            # Control speech duration
    apply_denoising=True         # Apply post-processing denoising
)
```

## Working with Different EEG Systems

### Adjusting for different EEG channel counts

```python
# For 32-channel EEG systems
eeg_processor = EEGProcessor(
    sampling_rate=256,
    channels=32,
    channel_map='configs/channel_maps/32ch_to_64ch.yaml'  # Channel mapping config
)
```

### Custom EEG Preprocessing

```python
# Custom preprocessing pipeline
eeg_processor = EEGProcessor(
    sampling_rate=256,
    channels=64,
    notch_freq=50.0,            # For 50Hz power line frequency
    highpass_freq=0.5,          # Custom high-pass filter
    lowpass_freq=80.0,          # Custom low-pass filter
    apply_ica=True,             # Apply Independent Component Analysis
    ica_components_to_reject=[0, 3, 5]  # Reject specific ICA components
)
```

## Speech Output Options

### Output Formats

```python
# Save as WAV file
speech_output.save_to_file('output.wav')

# Save as MP3 file
speech_output.save_to_file('output.mp3', format='mp3', bitrate='192k')

# Export as numpy array
speech_array = speech_output.to_numpy()
```

### Speech Enhancement

```python
# Apply post-processing for enhanced quality
enhanced_speech = speech_output.enhance(
    noise_reduction=0.6,
    equalizer_preset='voice',
    normalize=True
)
enhanced_speech.save_to_file('enhanced_output.wav')
```

## Performance Considerations

### Memory Usage

For large EEG files:

```python
# Process in chunks to reduce memory usage
eeg_processor = EEGProcessor(chunk_size=60)  # 60-second chunks
for chunk in eeg_processor.iter_chunks('large_eeg_file.edf'):
    speech_chunk = model.generate_speech(chunk)
    speech_chunk.save_to_file(f'output_{chunk.timestamp}.wav')
```

### Inference Speed

Tips for faster inference:
1. Use quantized models for 3-4x speedup
2. Process in batches where possible
3. Use GPU acceleration for large models
4. Reduce sampling rate when possible
5. Use simpler vocoder options for real-time applications

## Error Handling

```python
try:
    speech_output = model.generate_speech(processed_signal)
except ValueError as e:
    print(f"Invalid input: {e}")
    # Handle error appropriately
except RuntimeError as e:
    print(f"Inference error: {e}")
    # Consider falling back to CPU if GPU memory error
```

## Logging and Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Create processor with verbose output
eeg_processor = EEGProcessor(verbose=True)
```

## Next Steps

- [Evaluate the generated speech](evaluation.md) quality
- [Fine-tune the model](training.md) for specific subjects
- [Optimize the model](optimization.md) for your deployment environment
