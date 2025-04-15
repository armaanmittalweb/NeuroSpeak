import argparse
import os
import yaml
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union

from neurospeak.model import NeuroSpeakModel
from neurospeak.eeg_processor import EEGProcessor
from neurospeak.data_loader import EEGSpeechDataset
from neurospeak.utils.optimization import ModelOptimizer

def optimize_model(
    model_path: str,
    data_dir: str,
    output_dir: str,
    quantize: bool = True,
    prune: bool = False,
    to_onnx: bool = True,
    parallel: bool = True,
    num_workers: Optional[int] = None,
    batch_size: int = 32
):
    """
    Optimize a trained NeuroSpeak model.
    
    Args:
        model_path: Path to the trained model directory
        data_dir: Directory containing sample data for optimization
        output_dir: Directory to save optimized model
        quantize: Whether to apply quantization
        prune: Whether to apply pruning
        to_onnx: Whether to convert to ONNX format
        parallel: Whether to set up parallel inference
        num_workers: Number of worker processes for parallel inference
        batch_size: Batch size for optimization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    model = NeuroSpeakModel.from_pretrained(model_path)
    
    # Load sample data for optimization
    print(f"Loading sample data from {data_dir}...")
    dataset = EEGSpeechDataset(data_dir, batch_size=batch_size)
    representative_data = dataset.get_representative_samples(num_samples=100)
    
    # Create model optimizer
    optimizer = ModelOptimizer(model)
    
    # Apply optimizations
    print("Applying optimizations...")
    optimization_results = optimizer.optimize_for_inference(
        representative_data=representative_data['eeg_data'],
        quantize=quantize,
        prune=prune,
        training_dataset=dataset.get_train_dataset() if prune else None,
        to_onnx=to_onnx,
        parallel=parallel,
        num_workers=num_workers
    )
    
    # Save optimized model
    optimized_model = optimization_results['optimized_model']
    optimized_model.save_pretrained(os.path.join(output_dir, 'optimized_model'))
    
    # Save optimization report
    report = {
        'original_inference_time_ms': float(optimization_results['original_inference_time']),
        'optimized_inference_time_ms': float(optimization_results['optimized_inference_time']),
        'speedup_factor': float(optimization_results['speedup']),
        'optimizations_applied': {
            'quantization': quantize,
            'pruning': prune,
            'onnx_conversion': to_onnx,
            'parallel_processing': parallel
        }
    }
    
    report_path = os.path.join(output_dir, 'optimization_report.yaml')
    with open(report_path, 'w') as f:
        yaml.dump(report, f)
    
    print(f"Optimization completed! Report saved to {report_path}")
    print(f"Speedup achieved: {report['speedup_factor']:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize NeuroSpeak model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model directory')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing sample data')
    parser.add_argument('--output-dir', type=str, default='./optimized_model', help='Directory to save optimized model')
    parser.add_argument('--no-quantize', action='store_false', dest='quantize', help='Disable quantization')
    parser.add_argument('--prune', action='store_true', help='Enable pruning (requires retraining)')
    parser.add_argument('--no-onnx', action='store_false', dest='to_onnx', help='Disable ONNX conversion')
    parser.add_argument('--no-parallel', action='store_false', dest='parallel', help='Disable parallel inference')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for optimization')
    
    args = parser.parse_args()
    
    optimize_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        quantize=args.quantize,
        prune=args.prune,
        to_onnx=args.to_onnx,
        parallel=args.parallel,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )