"""
Framework-agnostic neural network model profiler for calculating FLOPs and parameters.
"""

# Standard imports
import os
import inspect
from typing import Dict, List, Optional, Union, Any, Tuple

from utils.calc_flops.utils.profile_exporter import ProfileExporter
from utils.calc_flops.utils.profile_reporter import ProfileReporter

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# PyTorch imports
try:
    import torch
    import numpy as np

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Optional PyTorch FLOP counters
try:
    from fvcore.nn import FlopCountAnalysis

    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

try:
    import ptflops

    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False


class ModelProfiler:
    """
    A framework-agnostic model profiler for calculating FLOPs, parameters, and other metrics.
    """

    def __init__(self, model: Any, sample_input: Optional[Any] = None, framework: str = 'auto'):
        """
        Initialize the model profiler

        Args:
            model: The neural network model
            sample_input: Sample input data for the model (optional, can be provided later)
            framework: Framework used ('tensorflow', 'pytorch', 'auto' to detect)
        """
        self.model = model
        self.sample_input = sample_input
        self.results = {}

        # Auto-detect or use specified framework
        if framework == 'auto':
            self.framework = self._detect_framework()
        else:
            self.framework = framework

        # Validate that necessary packages are available
        self._validate_environment()

    def _detect_framework(self) -> str:
        """Auto-detect which framework the model belongs to: 'tensorflow' or 'pytorch'"""

        # First check availability flags
        if PYTORCH_AVAILABLE and not TENSORFLOW_AVAILABLE:
            return 'pytorch'
        elif TENSORFLOW_AVAILABLE and not PYTORCH_AVAILABLE:
            return 'tensorflow'

        if hasattr(self.model, 'inputs'):
            if isinstance(self.model, tf.keras.Model):
                return 'tensorflow'
            if ((isinstance(self.model.inputs, list) and any(
                    isinstance(inp, tf.TensorSpec) for inp in self.model.inputs))
                    or isinstance(self.model.inputs, tf.TensorSpec)):
                return 'tensorflow'

        if hasattr(self.model, 'parameters') and callable(getattr(self.model, 'parameters', None)):
            return 'pytorch'

        module_name = inspect.getmodule(self.model.__class__).__name__
        if 'torch' in module_name.lower():
            return 'pytorch'
        elif any(fw in module_name.lower() for fw in ['tensorflow', 'keras']):
            return 'tensorflow'

        return 'pytorch'

    def _validate_environment(self) -> None:
        """Validate that the necessary packages are available"""
        if self.framework == 'tensorflow':
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required for profiling TensorFlow models")

            self.tf_version = tf.__version__

        elif self.framework == 'pytorch':
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch is required for profiling PyTorch models")

            self.torch_version = torch.__version__

            # Check for FLOP calculation packages
            self.fvcore_available = FVCORE_AVAILABLE

    def calculate_flops(self, sample_input: Optional[Any] = None,
                        ts_output_path: Optional[str] = None) -> 'ModelProfiler':
        """
        Calculate FLOPs for the model

        Args:
            sample_input: Sample input data (optional if already provided in constructor)
            ts_output_path: Optional path for TensorFlow profiler output

        Returns:
            self: Returns self for method chaining
        """
        if sample_input is not None:
            self.sample_input = sample_input

        if self.framework == 'tensorflow':
            self.results.update(self._calculate_flops_tensorflow(ts_output_path))
        elif self.framework == 'pytorch':
            if self.sample_input is None:
                raise ValueError("Sample input is required for PyTorch FLOPs calculation")
            self.results.update(self._calculate_flops_pytorch())
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        # Add framework info to results
        self.results['framework'] = self.framework
        self.results['model'] = self.model

        return self

    def _calculate_flops_tensorflow(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate FLOPs for TensorFlow models using convert_variables_to_constants_v2 and v1 profiler

        Args:
            output_path: Optional directory path to save profiler output

        Returns:
            dict: Results with FLOPs and parameter counts
        """

        try:
            # Configure profiler options
            opts = ProfileOptionBuilder.float_operation()

            # Set output path if specified
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                opts['output'] = f"file:outfile={os.path.join(output_path, 'flops_profile')}"
            else:
                opts['output'] = 'none'

            # Handle different input shapes
            if isinstance(self.model.input_shape, list):
                # Create appropriate input signature for multi-input models
                input_signatures = [tf.TensorSpec(shape=(1,) + x[1:]) for x in self.model.input_shape]
                forward_pass = tf.function(self.model.call)
                graph_info = profile(
                    forward_pass.get_concrete_function(input_signatures).graph,
                    options=opts
                )
            else:
                # Single input model
                forward_pass = tf.function(
                    self.model.call,
                    input_signature=[tf.TensorSpec(shape=(1,) + self.model.input_shape[1:])]
                )
                graph_info = profile(
                    forward_pass.get_concrete_function().graph,
                    options=opts
                )

            # Return results with parameter counts
            return {
                'total_flops': graph_info.total_float_ops,
                'params': self._count_tensorflow_params()
            }

        except Exception as e:
            return {'error': f"Error calculating TensorFlow FLOPs: {str(e)}"}

    def _count_tensorflow_params(self) -> Dict[str, int]:
        """Count parameters in a TensorFlow model"""
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        for var in self.model.variables:
            param_count = np.prod(var.shape)
            total_params += param_count

            if var.trainable:
                trainable_params += param_count
            else:
                non_trainable_params += param_count

        return {
            'total': int(total_params),
            'trainable': int(trainable_params),
            'non_trainable': int(non_trainable_params)
        }

    def _calculate_flops_pytorch(self) -> Dict[str, Any]:
        """Calculate FLOPs for PyTorch models"""
        if not self.fvcore_available:
            return {'error': "fvcore is required for PyTorch FLOPs calculation"}

        try:
            # Ensure the model is in eval mode
            self.model.eval()

            # Prepare inputs (use just the first sample if batch is provided)
            _inputs = self.sample_input
            if isinstance(_inputs, torch.Tensor):
                _inputs = _inputs[:1]
            elif isinstance(_inputs, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in _inputs):
                _inputs = [x[:1] for x in _inputs]

            # Calculate FLOPs using fvcore
            flops_counter = FlopCountAnalysis(self.model, _inputs)

            return {
                'total_flops': flops_counter.total() * 2,  # Multiply by 2 to count both mult and add
                'by_operator': flops_counter.by_operator(),
                'by_module': flops_counter.by_module(),
                'params': self._count_pytorch_params()
            }

        except Exception as e:
            return {'error': f"Error calculating PyTorch FLOPs: {str(e)}"}

    def _count_pytorch_params(self) -> Dict[str, int]:
        """Count parameters in a PyTorch model"""
        total_params = 0
        trainable_params = 0

        for param in self.model.parameters():
            num_params = param.numel()
            total_params += num_params

            if param.requires_grad:
                trainable_params += num_params

        return {
            'total': int(total_params),
            'trainable': int(trainable_params),
            'non_trainable': int(total_params - trainable_params)
        }

    def get_results(self) -> Dict[str, Any]:
        """Get the profiling results"""
        return self.results

    def print_report(self, units: str = 'G', detailed: bool = False) -> None:
        """
        Print a beautified report of the profiling results

        Args:
            units: Units to display ('G' for GFLOPs, 'M' for MFLOPs, 'K' for KFLOPs)
            detailed: Whether to print detailed breakdown by operator/module if available
        """
        # Use the ProfileReporter class internally
        reporter = ProfileReporter(self.results)
        reporter.print_report(units, detailed)

    def export_results(self, export_path: str, units: str = 'G', detailed: bool = False,
                       formats: Optional[List[str]] = None) -> None:
        """
        Export profiling results to files in specified formats

        Args:
            export_path: Path to folder where results will be saved
            units: Units to display ('G' for GFLOPs, 'M' for MFLOPs, 'K' for KFLOPs)
            detailed: Whether to include detailed breakdown
            formats: List of formats to export ('json', 'csv', 'txt', 'md', 'all')
                    If None, defaults to ['json', 'txt']
        """
        # Use the ProfileExporter class internally
        exporter = ProfileExporter(self.results)
        exporter.export(export_path, units, detailed, formats)
