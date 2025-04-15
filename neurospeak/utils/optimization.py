import tensorflow as tf
import numpy as np
import os
import time
import multiprocessing
from typing import Dict, List, Optional, Tuple, Union, Callable

class ModelOptimizer:
    """
    Optimizer for NeuroSpeak models.
    
    Implements various optimization techniques:
    - Quantization
    - Pruning
    - Knowledge distillation
    - Parallel processing
    - ONNX runtime conversion
    """
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize the model optimizer.
        
        Args:
            model: TensorFlow model to optimize
        """
        self.model = model
        self.optimized_model = None
        self.original_inference_time = None
        self.optimized_inference_time = None
        
    def benchmark(self, input_data: np.ndarray, num_runs: int = 50) -> float:
        """
        Benchmark model inference time.
        
        Args:
            input_data: Sample input data for benchmarking
            num_runs: Number of inference runs to average over
            
        Returns:
            Average inference time in milliseconds
        """
        # Warmup runs
        for _ in range(5):
            _ = self.model(input_data)
        
        # Benchmark runs
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.model(input_data)
        end_time = time.time()
        
        # Calculate average inference time
        avg_time_ms = (end_time - start_time) * 1000 / num_runs
        return avg_time_ms
    
    def quantize(self, representative_dataset: Callable) -> tf.keras.Model:
        """
        Apply post-training quantization to the model.
        
        Args:
            representative_dataset: Function that provides representative data for quantization
            
        Returns:
            Quantized model
        """
        print("Applying quantization to the model...")
        
        # Convert to TensorFlow Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset for quantization
        converter.representative_dataset = representative_dataset
        
        # Force full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Convert the model
        quantized_tflite_model = converter.convert()
        
        # Load the quantized model
        interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        
        # Create a wrapper model
        class QuantizedModel(tf.keras.Model):
            def __init__(self, interpreter):
                super().__init__()
                self.interpreter = interpreter
                self.input_details = interpreter.get_input_details()
                self.output_details = interpreter.get_output_details()
                
            def call(self, inputs):
                # Convert input to the required format
                input_data = tf.cast(inputs, tf.int8).numpy()
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                
                # Run inference
                self.interpreter.invoke()
                
                # Get output
                output = self.interpreter.get_tensor(self.output_details[0]['index'])
                return tf.convert_to_tensor(output)
        
        self.optimized_model = QuantizedModel(interpreter)
        print("Quantization completed successfully.")
        return self.optimized_model
    
    def apply_pruning(self, training_dataset, epochs: int = 10, target_sparsity: float = 0.5) -> tf.keras.Model:
        """
        Apply weight pruning to the model.
        
        Args:
            training_dataset: Dataset for pruning-aware training
            epochs: Number of pruning-aware training epochs
            target_sparsity: Target sparsity level (fraction of weights to prune)
            
        Returns:
            Pruned model
        """
        print(f"Applying pruning to the model (target sparsity: {target_sparsity})...")
        
        # Import pruning library
        import tensorflow_model_optimization as tfmot
        
        # Define pruning schedule
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=epochs * len(training_dataset)
        )
        
        # Apply pruning to all layers
        pruning_params = {
            'pruning_schedule': pruning_schedule,
            'block_size': (1, 1),
            'block_pooling_type': 'AVG'
        }
        
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model, **pruning_params
        )
        
        # Compile the pruned model with the same optimizer and loss
        pruned_model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        # Define pruning callbacks
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries()
        ]
        
        # Train the model with pruning
        pruned_model.fit(
            training_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Strip pruning wrapper for inference
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        
        self.optimized_model = final_model
        print("Pruning completed successfully.")
        return self.optimized_model
    
    def convert_to_onnx(self, output_path: str = 'model.onnx') -> str:
        """
        Convert TensorFlow model to ONNX format for faster inference.
        
        Args:
            output_path: Path to save the ONNX model
            
        Returns:
            Path to the saved ONNX model
        """
        try:
            # Check for tf2onnx
            import tf2onnx
            
            print(f"Converting model to ONNX format (output: {output_path})...")
            
            # Convert model to ONNX
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model,
                opset=13,
                output_path=output_path
            )
            
            print("ONNX conversion completed successfully.")
            return output_path
            
        except ImportError:
            print("Error: tf2onnx package not found. Please install it with: pip install tf2onnx")
            return None
    
    def create_parallel_inference(self, num_workers: int = None) -> Callable:
        """
        Create a function for parallel inference using multiple CPU cores.
        
        Args:
            num_workers: Number of worker processes (default: number of CPU cores)
            
        Returns:
            Function for parallel inference
        """
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
            
        print(f"Setting up parallel inference with {num_workers} workers...")
        
        # Define the worker function
        def worker_fn(model, input_queue, output_queue):
            while True:
                job_id, inputs = input_queue.get()
                if job_id is None:  # Poison pill
                    break
                    
                # Perform inference
                outputs = model.predict(inputs)
                output_queue.put((job_id, outputs))
        
        # Create model copy for each worker
        target_model = self.optimized_model if self.optimized_model else self.model
        
        # Create queues
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()
        
        # Start worker processes
        processes = []
        for _ in range(num_workers):
            process = multiprocessing.Process(
                target=worker_fn,
                args=(target_model, input_queue, output_queue)
            )
            process.daemon = True
            process.start()
            processes.append(process)
        
        # Create the parallel inference function
        def parallel_inference_fn(batch_inputs):
            # Split batch into chunks for workers
            batch_size = batch_inputs.shape[0]
            chunk_size = max(1, batch_size // num_workers)
            chunks = []
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunks.append(batch_inputs[i:end_idx])
            
            # Submit jobs
            results = {}
            for i, chunk in enumerate(chunks):
                input_queue.put((i, chunk))
            
            # Collect results
            for _ in range(len(chunks)):
                job_id, outputs = output_queue.get()
                results[job_id] = outputs
            
            # Combine results in correct order
            combined_outputs = np.concatenate([results[i] for i in range(len(chunks))], axis=0)
            return combined_outputs
        
        # Store processes for cleanup
        parallel_inference_fn.processes = processes
        parallel_inference_fn.queues = (input_queue, output_queue)
        
        # Add cleanup function
        def cleanup():
            # Send poison pills to workers
            for _ in processes:
                input_queue.put((None, None))
            
            # Wait for processes to terminate
            for p in processes:
                p.join()
                
        parallel_inference_fn.cleanup = cleanup
        
        print("Parallel inference setup completed.")
        return parallel_inference_fn
    
    def optimize_for_inference(
        self,
        representative_data: np.ndarray,
        quantize: bool = True,
        prune: bool = False,
        training_dataset = None,
        to_onnx: bool = False,
        parallel: bool = True,
        num_workers: int = None
    ) -> Dict[str, Union[tf.keras.Model, Callable, float]]:
        """
        Apply multiple optimization techniques to the model.
        
        Args:
            representative_data: Sample data for optimization
            quantize: Whether to apply quantization
            prune: Whether to apply pruning
            training_dataset: Dataset for pruning-aware training (required if prune=True)
            to_onnx: Whether to convert to ONNX format
            parallel: Whether to set up parallel inference
            num_workers: Number of worker processes for parallel inference
            
        Returns:
            Dictionary with optimization results
        """
        # Benchmark original model
        self.original_inference_time = self.benchmark(representative_data)
        print(f"Original model inference time: {self.original_inference_time:.2f} ms")
        
        # Apply optimizations
        if prune and training_dataset is not None:
            self.apply_pruning(training_dataset)
        
        if quantize:
            # Create representative dataset function for quantization
            def representative_dataset_fn():
                for i in range(representative_data.shape[0]):
                    yield [np.expand_dims(representative_data[i], axis=0)]
                    
            self.quantize(representative_dataset_fn)
        
        # Convert to ONNX if requested
        onnx_path = None
        if to_onnx:
            onnx_path = self.convert_to_onnx()
        
        # Set up parallel inference if requested
        parallel_fn = None
        if parallel:
            parallel_fn = self.create_parallel_inference(num_workers)
        
        # Benchmark optimized model
        target_model = self.optimized_model if self.optimized_model else self.model
        self.optimized_inference_time = self.benchmark(representative_data)
        print(f"Optimized model inference time: {self.optimized_inference_time:.2f} ms")
        
        # Calculate speedup
        speedup = self.original_inference_time / self.optimized_inference_time
        print(f"Speedup: {speedup:.2f}x")
        
        # Return optimization results
        return {
            'original_model': self.model,
            'optimized_model': target_model,
            'original_inference_time': self.original_inference_time,
            'optimized_inference_time': self.optimized_inference_time,
            'speedup': speedup,
            'onnx_path': onnx_path,
            'parallel_inference_fn': parallel_fn
        }
