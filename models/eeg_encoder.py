import numpy as np
import mne
from scipy import signal
from typing import Dict, List, Optional, Tuple, Union

class EEGProcessor:
    """
    Processor for EEG signals.
    
    Handles loading, preprocessing, and feature extraction from EEG data.
    """
    
    def __init__(
        self,
        sampling_rate: int = 256,
        channels: int = 64,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        notch_freq: float = 60.0,
        highpass_freq: float = 1.0,
        lowpass_freq: float = 70.0
    ):
        """
        Initialize the EEG processor.
        
        Args:
            sampling_rate: Sampling rate of the EEG signal in Hz
            channels: Number of EEG channels
            freq_bands: Frequency bands for feature extraction
            notch_freq: Frequency to remove (usually power line noise)
            highpass_freq: High-pass filter cutoff
            lowpass_freq: Low-pass filter cutoff
        """
        self.sampling_rate = sampling_rate
        self.channels = channels
        
        # Default frequency bands if not provided
        self.freq_bands = freq_bands or {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 70)
        }
        
        self.notch_freq = notch_freq
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
    
    def load_data(self, file_path: str) -> np.ndarray:
        """
        Load EEG data from file.
        
        Args:
            file_path: Path to the EEG data file
            
        Returns:
            EEG data as numpy array
        """
        # Determine file format and load accordingly
        if file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(file_path, preload=True)
        elif file_path.endswith('.bdf'):
            raw = mne.io.read_raw_bdf(file_path, preload=True)
        elif file_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(file_path, preload=True)
        elif file_path.endswith('.npy'):
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Extract data as numpy array
        data = raw.get_data()
        
        return data
    
    def preprocess(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG data.
        
        Steps:
        1. Filtering (notch, bandpass)
        2. Artifact removal
        3. Normalization
        
        Args:
            eeg_data: Raw EEG data
            
        Returns:
            Preprocessed EEG data
        """
        # Make sure data is 2D (channels x time)
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        # Apply notch filter to remove power line noise
        notch_b, notch_a = signal.iirnotch(
            self.notch_freq, 
            Q=30, 
            fs=self.sampling_rate
        )
        eeg_data = signal.filtfilt(notch_b, notch_a, eeg_data, axis=1)
        
        # Apply bandpass filter
        b, a = signal.butter(
            N=4, 
            Wn=[self.highpass_freq, self.lowpass_freq], 
            btype='bandpass', 
            fs=self.sampling_rate
        )
        eeg_data = signal.filtfilt(b, a, eeg_data, axis=1)
        
        # Simple artifact removal (threshold-based)
        # In a real application, more sophisticated methods would be used
        threshold = 5 * np.std(eeg_data)
        eeg_data = np.clip(eeg_data, -threshold, threshold)
        
        # Z-score normalization
        eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / \
                   (np.std(eeg_data, axis=1, keepdims=True) + 1e-10)
        
        return eeg_data
    
    def extract_features(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from preprocessed EEG data.
        
        Features include:
        - Power spectral density for each frequency band
        - Time-domain statistics
        
        Args:
            eeg_data: Preprocessed EEG data
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Compute power spectral density
        freqs, psd = signal.welch(
            eeg_data, 
            fs=self.sampling_rate, 
            nperseg=min(256, eeg_data.shape[1])
        )
        
        # Extract band powers
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            band_power = np.mean(psd[:, idx_band], axis=1)
            features[f"{band_name}_power"] = band_power
        
        # Time-domain features
        features['mean'] = np.mean(eeg_data, axis=1)
        features['std'] = np.std(eeg_data, axis=1)
        features['kurtosis'] = scipy.stats.kurtosis(eeg_data, axis=1)
        features['hjorth_mobility'] = self._hjorth_mobility(eeg_data)
        features['hjorth_complexity'] = self._hjorth_complexity(eeg_data)
        
        return features
    
    def _hjorth_mobility(self, data: np.ndarray) -> np.ndarray:
        """Calculate Hjorth mobility parameter."""
        # First derivative variance
        diff1 = np.diff(data, axis=1)
        var_diff1 = np.var(diff1, axis=1)
        
        # Original signal variance
        var_data = np.var(data, axis=1)
        
        return np.sqrt(var_diff1 / var_data)
    
    def _hjorth_complexity(self, data: np.ndarray) -> np.ndarray:
        """Calculate Hjorth complexity parameter."""
        # First derivative
        diff1 = np.diff(data, axis=1)
        
        # Second derivative
        diff2 = np.diff(diff1, axis=1)
        
        # Variances
        var_diff1 = np.var(diff1, axis=1)
        var_diff2 = np.var(diff2, axis=1)
        
        # Mobility of first derivative
        mob_diff1 = np.sqrt(var_diff2 / var_diff1)
        
        # Mobility of original signal
        mob_data = self._hjorth_mobility(data)
        
        return mob_diff1 / mob_data