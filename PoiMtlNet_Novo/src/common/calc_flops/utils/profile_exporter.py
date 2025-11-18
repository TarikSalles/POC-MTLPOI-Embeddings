import csv
import datetime
import json
import os
from typing import Any, Dict, List, Optional


class ProfileExporter:
    """
    Exporter class for saving model profiling results to various file formats
    """

    def __init__(self, profiler_results: Dict[str, Any]):
        """
        Initialize the exporter with profiling results

        Args:
            profiler_results: Results dictionary from ModelProfiler
        """
        self.results = profiler_results

    def export(self, export_path: str, units: str = 'G', detailed: bool = False,
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
        # Create directory if it doesn't exist
        os.makedirs(export_path, exist_ok=True)

        # Default formats if none specified
        if formats is None:
            formats = ['json', 'txt']
        elif 'all' in formats:
            formats = ['json', 'csv', 'txt', 'md']

        # Get timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = getattr(self.results.get('model', {}), 'name', 'model')
        base_filename = f"{model_name}_profile_{timestamp}"

        # Prepare data with proper units
        export_data = self._prepare_export_data(units, detailed)

        # Export in each requested format
        for fmt in formats:
            fmt = fmt.lower()
            if fmt == 'json':
                self._export_json(export_path, base_filename, export_data)
            elif fmt == 'csv':
                self._export_csv(export_path, base_filename, export_data)
            elif fmt == 'txt':
                self._export_txt(export_path, base_filename, export_data, units)
            elif fmt == 'md':
                self._export_markdown(export_path, base_filename, export_data, units)

    def _prepare_export_data(self, units: str, detailed: bool) -> Dict[str, Any]:
        """Prepare data for export with proper units"""
        # Get unit multiplier and name
        multipliers = {
            'G': 1e-9,
            'M': 1e-6,
            'K': 1e-3,
            'none': 1
        }

        unit_names = {
            'G': 'GFLOPs',
            'M': 'MFLOPs',
            'K': 'KFLOPs',
            'none': 'FLOPs'
        }

        multiplier = multipliers.get(units, 1e-9)
        unit_name = unit_names.get(units, 'GFLOPs')

        # Basic export data
        export_data = {
            'framework': self.results.get('framework', 'unknown'),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'units': unit_name
        }

        # Add computation metrics
        if 'total_flops' in self.results:
            export_data['total_flops_raw'] = self.results['total_flops']
            export_data['total_flops'] = round(self.results['total_flops'] * multiplier, 2)

        # Add parameter metrics
        if 'params' in self.results:
            if isinstance(self.results['params'], dict):
                export_data['parameters'] = self.results['params']
            else:
                export_data['parameters'] = {'total': self.results['params']}

        # Add detailed breakdown if requested
        if detailed:
            if 'by_operator' in self.results:
                export_data['by_operator'] = {
                    op: round(count * multiplier, 2)
                    for op, count in self.results['by_operator'].items()
                }

            if 'by_module' in self.results:
                export_data['by_module'] = {
                    mod: round(count * multiplier, 2)
                    for mod, count in self.results['by_module'].items()
                }

        return export_data

    def _export_json(self, export_path: str, base_filename: str, export_data: Dict[str, Any]) -> None:
        """Export results as JSON"""
        json_path = os.path.join(export_path, f"{base_filename}.json")
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✅ Exported JSON report to: {json_path}")

    def _export_csv(self, export_path: str, base_filename: str, export_data: Dict[str, Any]) -> None:
        """Export results as CSV (flattened structure)"""
        csv_path = os.path.join(export_path, f"{base_filename}.csv")

        # Flatten nested dictionaries for CSV
        flat_data = {}

        def flatten_dict(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten_dict(v, f"{prefix}{k}.")
                else:
                    flat_data[f"{prefix}{k}"] = v

        flatten_dict(export_data)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for k, v in sorted(flat_data.items()):
                writer.writerow([k, v])

        print(f"✅ Exported CSV report to: {csv_path}")

    def _export_txt(self, export_path: str, base_filename: str,
                    export_data: Dict[str, Any], units: str) -> None:
        """Export results as plain text (similar to print_report output)"""
        txt_path = os.path.join(export_path, f"{base_filename}.txt")

        with open(txt_path, 'w') as f:
            # Header section
            f.write(f"{'=' * 60}\n")
            f.write(f"Model Profiling Report ({export_data.get('framework', 'Unknown').capitalize()} model)\n")
            f.write(f"Generated: {export_data.get('timestamp')}\n")
            f.write(f"{'=' * 60}\n\n")

            self._write_txt_flops_section(f, export_data, units)
            self._write_txt_params_section(f, export_data)
            self._write_txt_detailed_section(f, export_data, units)

        print(f"✅ Exported TXT report to: {txt_path}")

    def _write_txt_flops_section(self, file, export_data: Dict[str, Any], units: str) -> None:
        """Write FLOPs section to text file"""
        if 'total_flops' in export_data:
            total = export_data['total_flops']
            file.write(f"Computational Complexity\n")
            file.write(f"  - Total: {total} {export_data.get('units')}\n")

            if units == 'G' and total >= 1000:
                file.write(f"  - Equivalent to {total / 1000:.2f} TFLOPs\n")

    def _write_txt_params_section(self, file, export_data: Dict[str, Any]) -> None:
        """Write parameters section to text file"""
        if 'parameters' in export_data:
            params = export_data['parameters']

            if isinstance(params, dict) and 'total' in params:
                total_params = params['total']

                file.write(f"\nModel Parameters\n")

                if total_params >= 1e9:
                    file.write(f"  - Total: {total_params / 1e9:.2f} B\n")
                elif total_params >= 1e6:
                    file.write(f"  - Total: {total_params / 1e6:.2f} M\n")
                elif total_params >= 1e3:
                    file.write(f"  - Total: {total_params / 1e3:.2f} K\n")
                else:
                    file.write(f"  - Total: {total_params}\n")

                if 'trainable' in params:
                    trainable = params['trainable']
                    file.write(f"  - Trainable: {trainable / 1e6:.2f} M ({trainable / total_params * 100:.1f}%)\n")

                if 'non_trainable' in params:
                    non_trainable = params['non_trainable']
                    file.write(
                        f"  - Non-trainable: {non_trainable / 1e6:.2f} M ({non_trainable / total_params * 100:.1f}%)\n")
            else:
                # Handle case where params is just a number
                if params >= 1e9:
                    file.write(f"\nParameters: {params / 1e9:.2f} B\n")
                elif params >= 1e6:
                    file.write(f"\nParameters: {params / 1e6:.2f} M\n")
                else:
                    file.write(f"\nParameters: {params}\n")

    def _write_txt_detailed_section(self, file, export_data: Dict[str, Any], units: str) -> None:
        """Write detailed breakdown section to text file"""
        if 'by_operator' in export_data or 'by_module' in export_data:
            file.write(f"\nDetailed Breakdown\n")
            file.write(f"{'-' * 60}\n")

            if 'by_operator' in export_data:
                file.write("By Operator Type:\n")
                operators = export_data['by_operator']
                sorted_ops = sorted(operators.items(), key=lambda x: x[1], reverse=True)

                for op, count in sorted_ops:
                    if count >= 0.01:  # Only show significant contributors
                        total_flops = export_data.get('total_flops_raw', 1)
                        percentage = count / total_flops * 100 if total_flops else 0
                        file.write(f"  {op:30} : {count:10.2f} {export_data.get('units')} ({percentage:5.1f}%)\n")

            if 'by_module' in export_data:
                file.write("\nBy Module (Top 10):\n")
                modules = export_data['by_module']
                sorted_modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)[:10]

                for module, count in sorted_modules:
                    if count >= 0.01:  # Only show significant contributors
                        # Truncate very long module names
                        if len(module) > 50:
                            module = module[:47] + "..."

                        total_flops = export_data.get('total_flops_raw', 1)
                        percentage = count / total_flops * 100 if total_flops else 0
                        file.write(f"  {module:50} : {count:10.2f} {export_data.get('units')} ({percentage:5.1f}%)\n")

    def _export_markdown(self, export_path: str, base_filename: str,
                         export_data: Dict[str, Any], units: str) -> None:
        """Export results as Markdown"""
        md_path = os.path.join(export_path, f"{base_filename}.md")

        with open(md_path, 'w') as f:
            # Header
            f.write(f"# Model Profiling Report\n\n")
            f.write(f"- **Framework**: {export_data.get('framework', 'Unknown').capitalize()}\n")
            f.write(f"- **Generated**: {export_data.get('timestamp')}\n")
            f.write(f"- **Model**: {base_filename.split('_profile_')[0]}\n\n")

            # FLOPs
            if 'total_flops' in export_data:
                total = export_data['total_flops']
                f.write(f"## Computational Complexity\n\n")
                f.write(f"- **Total**: {total} {export_data.get('units')}\n")

                if units == 'G' and total >= 1000:
                    f.write(f"- **Equivalent to**: {total / 1000:.2f} TFLOPs\n")

            # Parameters
            if 'parameters' in export_data:
                params = export_data['parameters']
                f.write(f"\n## Model Parameters\n\n")

                if isinstance(params, dict) and 'total' in params:
                    total_params = params['total']

                    if total_params >= 1e9:
                        f.write(f"- **Total**: {total_params / 1e9:.2f} B\n")
                    elif total_params >= 1e6:
                        f.write(f"- **Total**: {total_params / 1e6:.2f} M\n")
                    elif total_params >= 1e3:
                        f.write(f"- **Total**: {total_params / 1e3:.2f} K\n")
                    else:
                        f.write(f"- **Total**: {total_params}\n")

                    if 'trainable' in params:
                        trainable = params['trainable']
                        f.write(f"- **Trainable**: {trainable / 1e6:.2f} M ({trainable / total_params * 100:.1f}%)\n")

                    if 'non_trainable' in params:
                        non_trainable = params['non_trainable']
                        f.write(
                            f"- **Non-trainable**: {non_trainable / 1e6:.2f} M ({non_trainable / total_params * 100:.1f}%)\n")
                else:
                    # Handle case where params is just a number
                    if params >= 1e9:
                        f.write(f"- **Parameters**: {params / 1e9:.2f} B\n")
                    elif params >= 1e6:
                        f.write(f"- **Parameters**: {params / 1e6:.2f} M\n")
                    else:
                        f.write(f"- **Parameters**: {params}\n")

            # Detailed breakdown
            if 'by_operator' in export_data or 'by_module' in export_data:
                f.write(f"\n## Detailed Breakdown\n\n")

                if 'by_operator' in export_data:
                    f.write("### By Operator Type\n\n")
                    f.write("| Operator | FLOPs | Percentage |\n")
                    f.write("|----------|-------|------------|\n")

                    operators = export_data['by_operator']
                    sorted_ops = sorted(operators.items(), key=lambda x: x[1], reverse=True)

                    for op, count in sorted_ops:
                        if count >= 0.01:  # Only show significant contributors
                            total_flops = export_data.get('total_flops_raw', 1)
                            percentage = count / total_flops * 100 if total_flops else 0
                            f.write(f"| {op} | {count} {export_data.get('units')} | {percentage:.1f}% |\n")

                if 'by_module' in export_data:
                    f.write("\n### By Module (Top 10)\n\n")
                    f.write("| Module | FLOPs | Percentage |\n")
                    f.write("|--------|-------|------------|\n")

                    modules = export_data['by_module']
                    sorted_modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)[:10]

                    for module, count in sorted_modules:
                        if count >= 0.01:  # Only show significant contributors
                            # Truncate very long module names
                            if len(module) > 40:
                                module = module[:37] + "..."

                            total_flops = export_data.get('total_flops_raw', 1)
                            percentage = count / total_flops * 100 if total_flops else 0
                            f.write(f"| {module} | {count} {export_data.get('units')} | {percentage:.1f}% |\n")

        print(f"✅ Exported Markdown report to: {md_path}")