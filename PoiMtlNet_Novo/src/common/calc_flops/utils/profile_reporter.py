from typing import Dict, Any, Tuple


class ProfileReporter:
    """
    Reporter class for displaying model profiling results in a user-friendly format
    """

    def __init__(self, profiler_results: Dict[str, Any]):
        """
        Initialize the reporter with profiling results

        Args:
            profiler_results: Results dictionary from ModelProfiler
        """
        self.results = profiler_results

    def print_report(self, units: str = 'G', detailed: bool = False) -> None:
        """
        Print a beautified report of the profiling results

        Args:
            units: Units to display ('G' for GFLOPs, 'M' for MFLOPs, 'K' for KFLOPs)
            detailed: Whether to print detailed breakdown by operator/module if available
        """
        # Handle case where no results available
        if not self.results:
            print("⚠️  No profiling results available. Run calculate_flops() first.")
            return

        # Handle errors
        if 'error' in self.results:
            print(f"❌ Error: {self.results['error']}")
            return

        # Get the multiplier and unit name based on selected units
        multiplier, unit_name = self._get_unit_multiplier(units)

        # Print header with framework info
        print(f"\n{'=' * 60}")
        print(f"🔍 Model Profiling Report ({self.results.get('framework', 'Unknown').capitalize()} model)")
        print(f"{'=' * 60}")

        # Print total FLOPs
        self._print_flops_section(multiplier, unit_name)

        # Print parameters if available
        self._print_params_section()

        # Print detailed breakdown if requested
        if detailed:
            self._print_detailed_breakdown(multiplier, unit_name)

        print(f"{'=' * 60}")

    def _get_unit_multiplier(self, units: str) -> Tuple[float, str]:
        """Convert unit selection to multiplier and name"""
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

        return multipliers.get(units, 1e-9), unit_names.get(units, 'GFLOPs')

    def _print_flops_section(self, multiplier: float, unit_name: str) -> None:
        """Print the FLOPs section of the report"""
        if 'total_flops' in self.results:
            total = self.results['total_flops'] * multiplier
            print(f"📊 Computational Complexity")
            print(f"  - Total: {total:.2f} {unit_name}")

            # Convert to human-readable format for large models
            if unit_name == 'GFLOPs' and total >= 1000:
                print(f"  - Equivalent to {total / 1000:.2f} TFLOPs")

    def _print_params_section(self) -> None:
        """Print the parameters section of the report"""
        if 'params' in self.results:
            params = self.results['params']

            if isinstance(params, dict) and 'total' in params:
                total_params = params['total']

                print(f"\n📏 Model Parameters")

                # Format total parameters
                if total_params >= 1e9:
                    print(f"  - Total: {total_params / 1e9:.2f} B")
                elif total_params >= 1e6:
                    print(f"  - Total: {total_params / 1e6:.2f} M")
                elif total_params >= 1e3:
                    print(f"  - Total: {total_params / 1e3:.2f} K")
                else:
                    print(f"  - Total: {total_params}")

                # Show trainable parameters if available
                if 'trainable' in params:
                    trainable = params['trainable']
                    if trainable >= 1e6:
                        print(f"  - Trainable: {trainable / 1e6:.2f} M ({trainable / total_params * 100:.1f}%)")
                    else:
                        print(f"  - Trainable: {trainable} ({trainable / total_params * 100:.1f}%)")

                # Show non-trainable parameters if available
                if 'non_trainable' in params:
                    non_trainable = params['non_trainable']
                    if non_trainable >= 1e6:
                        print(
                            f"  - Non-trainable: {non_trainable / 1e6:.2f} M ({non_trainable / total_params * 100:.1f}%)")
                    else:
                        print(f"  - Non-trainable: {non_trainable} ({non_trainable / total_params * 100:.1f}%)")
            else:
                # Handle case where params is just a number
                if params >= 1e9:
                    print(f"\n📏 Parameters: {params / 1e9:.2f} B")
                elif params >= 1e6:
                    print(f"\n📏 Parameters: {params / 1e6:.2f} M")
                else:
                    print(f"\n📏 Parameters: {params}")

    def _print_detailed_breakdown(self, multiplier: float, unit_name: str) -> None:
        """Print the detailed breakdown section of the report"""
        print(f"\n🔬 Detailed Breakdown")
        print(f"{'-' * 60}")

        # Print breakdown by operator type
        if 'by_operator' in self.results:
            self._print_operator_breakdown(multiplier, unit_name)

        # Print breakdown by module
        if 'by_module' in self.results:
            self._print_module_breakdown(multiplier, unit_name)

    def _print_operator_breakdown(self, multiplier: float, unit_name: str) -> None:
        """Print breakdown by operator type"""
        print("By Operator Type:")
        operators = self.results['by_operator']
        sorted_ops = sorted(operators.items(), key=lambda x: x[1], reverse=True)

        # Get the width for formatting
        max_width = max(len(op) for op, _ in sorted_ops) if sorted_ops else 10

        for op, count in sorted_ops:
            if count * multiplier >= 0.01:  # Only show significant contributors
                percentage = count / self.results['total_flops'] * 100
                print(f"  {op:{max_width}} : {count * multiplier:10.2f} {unit_name} ({percentage:5.1f}%)")

    def _print_module_breakdown(self, multiplier: float, unit_name: str) -> None:
        """Print breakdown by module"""
        print("\nBy Module (Top 10):")
        modules = self.results['by_module']
        sorted_modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)[:10]

        # Get the width for formatting
        max_width = max(len(mod) for mod, _ in sorted_modules) if sorted_modules else 10
        max_width = min(max_width, 50)  # Limit to 50 chars

        for module, count in sorted_modules:
            if count * multiplier >= 0.01:  # Only show significant contributors
                # Truncate very long module names
                if len(module) > max_width:
                    module = module[:max_width - 3] + "..."

                percentage = count / self.results['total_flops'] * 100
                print(f"  {module:{max_width}} : {count * multiplier:10.2f} {unit_name} ({percentage:5.1f}%)")