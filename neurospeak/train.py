import tensorflow as tf
import argparse
import yaml
import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

from neurospeak.model import NeuroSpeakModel
from neurospeak.eeg_processor import EEGProcessor
from neurospeak.data_loader import EEGSpeechDataset

def train(
    config_path: Optional[str] = None,
    data_dir: str = './data',
    output_dir: str = './trained_model',
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    validation_split: float = 0.2
):
    """
    Train a NeuroSpeak model.
    
    Args:
        config_path: Path to configuration YAML file
        data_dir: Directory containing training data
        output_dir: Directory to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        early_stopping_patience: Patience for early stopping
        validation_split: Fraction of data to use for validation
    """
    # Load configuration if provided
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override function arguments with config values
        data_dir = config.get('data_dir', data_dir)
        output_dir = config.get('output_dir', output_dir)
        epochs = config.get('epochs', epochs)
        batch_size = config.get('batch_size', batch_size)
        learning_rate = config.get('learning_rate', learning_rate)
        early_stopping_patience = config.get('early_stopping_patience', early_stopping_patience)
        validation_split = config.get('validation_split', validation_split)
        
        # Model configuration
        model_config = config.get('model', {})
    else:
        model_config = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_dir = os.path.join(output_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = EEGSpeechDataset(data_dir, batch_size=batch_size)
    train_dataset, val_dataset = dataset.get_train_val_split(validation_split)
    
    # Get dataset properties
    input_shape = dataset.get_eeg_shape()
    eeg_channels = input_shape[1]  # Assuming shape is (time_steps, channels)
    
    # Create model
    print("Creating NeuroSpeak model...")
    model = NeuroSpeakModel(
        input_shape=input_shape,
        eeg_channels=eeg_channels,
        **model_config
    )
    
    # Define optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss={
            'mel_output': tf.keras.losses.MeanSquaredError()
        },
        metrics={
            'mel_output': [
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanSquaredError()
            ]
        }
    )
    
    # Define callbacks
    callbacks = [
        tensorboard_callback,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'checkpoints', 'model_{epoch:02d}_{val_loss:.4f}'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=early_stopping_patience,
            monitor='val_loss',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    # Save trained model
    print(f"Training completed. Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.npy')
    np.save(history_path, history.history)
    
    print("Model training and saving completed successfully!")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NeuroSpeak model')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='./trained_model', help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Initial learning rate')
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )