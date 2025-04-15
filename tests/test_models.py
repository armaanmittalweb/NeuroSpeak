import unittest
import numpy as np
import tensorflow as tf
from neurospeak.model import NeuroSpeakModel

class TestNeuroSpeakModel(unittest.TestCase):
    """Test cases for NeuroSpeak model."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a small model for testing
        self.input_shape = (128, 64)  # 128 time steps, 64 channels
        self.model = NeuroSpeakModel(
            input_shape=self.input_shape,
            eeg_channels=64,
            rnn_units=64,  # Small size for testing
            transformer_layers=2,
            transformer_heads=4,
            transformer_dim=128,
            dropout_rate=0.1
        )
        
        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss={
                'mel_output': tf.keras.losses.MeanSquaredError()
            }
        )
    
    def test_model_initialization(self):
        """Test model is initialized correctly."""
        self.assertIsInstance(self.model, NeuroSpeakModel)
        self.assertEqual(self.model.input_shape, self.input_shape)
        self.assertEqual(self.model.eeg_channels, 64)
        
    def test_model_call(self):
        """Test forward pass."""
        # Create a dummy input
        batch_size = 2
        dummy_input = np.random.normal(size=(batch_size, *self.input_shape))
        
        # Run forward pass
        output = self.model(dummy_input)
        
        # Check output shape (batch_size, time_steps, mel_bands)
        # Time steps will be reduced due to pooling
        expected_time_steps = self.input_shape[0] // 4  # Due to two max pooling layers
        self.assertEqual(output.shape, (batch_size, expected_time_steps, 80))
    
    def test_save_load(self):
        """Test saving and loading the model."""
        import tempfile
        import os
        import shutil
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
    # Save the model
        self.model.save_pretrained(temp_dir)
        
        # Check if files exist
        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'config.json')))
        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'weights.h5')))
            
        # Load the model
        loaded_model = NeuroSpeakModel.from_pretrained(temp_dir)
            
        # Check if it's a NeuroSpeakModel
        self.assertIsInstance(loaded_model, NeuroSpeakModel)
            
        # Compare outputs
        dummy_input = np.random.normal(size=(2, *self.input_shape))
        original_output = self.model(dummy_input).numpy()