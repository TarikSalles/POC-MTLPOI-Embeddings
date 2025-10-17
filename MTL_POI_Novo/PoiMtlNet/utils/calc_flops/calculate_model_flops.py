from utils.calc_flops.model_profiler import ModelProfiler


def calculate_model_flops(model, sample_input=None, framework='auto', ts_output_path=None, print_report=False,
                          units='G', detailed=False, output_path=None, export_formats=None):
    """
    Calculate FLOPs for a deep learning model (TensorFlow or PyTorch)

    Args:
        model: The neural network model
        sample_input: Sample input data for the model
        framework: Framework used ('tensorflow', 'pytorch', or 'auto' to detect)
        ts_output_path: Optional path for TensorFlow profiler output files
        print_report: Whether to print a formatted report
        units: Units to display ('G' for GFLOPs, 'M' for MFLOPs, 'K' for KFLOPs)
        detailed: Whether to print detailed breakdown
        output_path: Path to folder for exporting results (None = no export)
        export_formats: List of formats to export ('json', 'csv', 'txt', 'md', 'all')

    Returns:
        dict: Dictionary with profiling results
    """
    profiler = ModelProfiler(model, sample_input, framework)
    profiler.calculate_flops(ts_output_path=ts_output_path)

    if print_report:
        profiler.print_report(units=units, detailed=detailed)

    if output_path:
        profiler.export_results(output_path, units=units, detailed=detailed, formats=export_formats)

    return profiler.results
