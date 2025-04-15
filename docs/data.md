# NeuroSpeak Data Guide

This document describes data requirements, formats, and preprocessing for NeuroSpeak models.

## Data Requirements

NeuroSpeak requires paired EEG and speech data:

1. **EEG Data**: Raw or preprocessed EEG recordings
2. **Speech Data**: Corresponding speech recordings or transcriptions

## Supported Data Formats

### EEG Data Formats
- European Data Format (EDF/BDF)
- MATLAB (.mat) files
- MNE-Python compatible formats
- NumPy (.npy) arrays

### Speech Data Formats
- WAV audio files (16kHz, 16-bit PCM recommended)
- MEL spectrograms (.npy)
- Phoneme/text transcriptions (.txt)

## Dataset Structure

The expected directory structure:

```
data/
├── subject_001/
│   ├── eeg/
│   │   ├── session_1.edf
│   │   └── session_2.edf
│   └── speech/
│       ├── session_1.wav
│       └── session_2.wav
├── subject_002/
│   ├── eeg/
│   └── speech/
└── metadata.csv
```

The `metadata.csv` file should contain:
- Subject ID
- Session ID
- EEG file path
- Speech file path
- Timestamps for alignment
- Additional metadata (e.g., task type, conditions)

## Data Preprocessing

### EEG Preprocessing
1. **Filtering**: Apply bandpass (1-70 Hz) and notch (50/60 Hz) filters
2. **Artifact Removal**: Remove eye blinks, muscle artifacts, etc.
3. **Re-referencing**: Convert to common average reference
4. **Downsampling**: Standardize to 256 Hz sampling rate
5. **Normalization**: Apply z-score normalization per channel

### Speech Preprocessing
1. **Resampling**: Convert to 16 kHz
2. **Feature Extraction**: Convert to mel spectrograms (80 bands)
3. **Normalization**: Apply mean/variance normalization
4. **Alignment**: Ensure proper time-alignment with EEG data

## Data Preparation Script

Use our data preparation script for automatic preprocessing:

```bash
python -m neurospeak.data.prepare_dataset \
    --eeg-dir /path/to/raw/eeg \
    --speech-dir /path/to/raw/speech \
    --output-dir ./processed_data \
    --sampling-rate 256 \
    --apply-filters \
    --remove-artifacts
```

## Training/Validation/Test Split

We recommend:
- 70% training
- 15% validation
- 15% test

For subject-independent evaluation, use leave-one-subject-out validation.

## Data Augmentation

For improved model generalization, consider:
1. **Time shifts**: ±20ms shifts
2. **Frequency masking**: Random frequency band masking
3. **Noise addition**: Low-level Gaussian noise
4. **Channel dropout**: Random EEG channel masking

Enable augmentation during training:

```bash
python -m neurospeak.train --data-dir ./data --output-dir ./model --augmentation configs/augmentation.yaml
```

## Public Datasets

NeuroSpeak is compatible with these public EEG datasets (additional preprocessing required):

1. **Temple University EEG Corpus**: Large clinical EEG dataset
2. **EEGManyPipelines**: Open EEG dataset with various cognitive tasks
3. **BCI Competition Datasets**: Motor imagery EEG recordings

## Creating Your Own Dataset

Guidelines for collecting high-quality EEG-speech data:
1. Use research-grade EEG equipment (64+ channels preferred)
2. Record speech in quiet environments with professional microphones
3. Ensure precise temporal synchronization
4. Include diverse speech tasks (single words, sentences, free speech)
5. Collect data from diverse participants

## Using Pretrained Models

For dataset compatibility with pretrained models:
1. Match preprocessing steps with training data
2. Use identical channel configurations
3. Apply the same normalization procedures

## Troubleshooting Data Issues

Common data issues and solutions:
- **Misalignment**: Use cross-correlation to align EEG and speech
- **Class imbalance**: Apply sample weighting
- **Noise artifacts**: Use robust artifact rejection methods
- **Inconsistent sampling**: Resample to standard rates

## Data Quality Metrics

Monitor these metrics to ensure data quality:
- Signal-to-noise ratio (SNR)
- Electrode impedance values (if available)
- Artifact percentage
- Alignment accuracy

## Next Steps

After preparing your dataset:
- [Train a model](training.md) using your data
- [Evaluate performance](evaluation.md) on test set
- [Optimize the model](optimization.md) for deployment
