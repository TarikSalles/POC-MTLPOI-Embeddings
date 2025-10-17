import json
import logging
from statistics import mean, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.ml_history.metrics import MLHistory

# 1) Configure the root logger with a sane default formatter
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 2) Create a custom formatter that only applies colour to our class's logger
class ClassColorFormatter(logging.Formatter):
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def format(self, record):
        msg = super().format(record)
        if record.name == HistoryDisplay.LOGGER_NAME:
            return f"{self.CYAN}{self.BOLD}{msg}{self.RESET}"
        return msg

# 3) Create and configure the HistoryDisplay logger
class HistoryDisplay:
    LOGGER_NAME = "ml.history.display"

    def __init__(self, history: "MLHistory"):
        self.h = history
        self.log = logging.getLogger(self.LOGGER_NAME)

        # ensure we only add our handler once
        if not any(isinstance(h, logging.StreamHandler) for h in self.log.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(
                ClassColorFormatter(
                    "%(asctime)s - %(levelname)-5s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.log.addHandler(handler)
            self.log.propagate = False

    def _sep(self, title: str, width: int = 60, sep: str = "=") -> str:
        """Generate a centred separator line."""
        pad = max(0, (width - len(title) - 2) // 2)
        return f"{sep * pad} {title} {sep * pad}"

    def _time_stats(self):
        elapsed_total = self.h.timer.timer()
        last_fold = self.h.folds[self.h.curr_i_fold - 1].timer.get_duration()
        past = [f.timer.get_duration() for f in self.h.folds[: self.h.curr_i_fold]]
        remaining = (mean(past) if past else 0) * (self.h.num_folds - self.h.curr_i_fold)
        return elapsed_total, last_fold, remaining

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds or minutes depending on the value."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes = seconds / 60
        return f"{minutes:.2f}m"

    def _format_metric(self, value, is_percentage: bool = True, width: int = 8) -> str:
        """Format a metric value, handling both numeric and pre-formatted string values."""
        if isinstance(value, (int, float)):
            if is_percentage:
                return f"{value * 100:>{width}.2f}%"
            return f"{value:>{width}}"
        # already a formatted string
        return f"{value:>{width}}"

    def start_fold(self):
        """Log the start of a fold."""
        text = f"FOLD {self.h.curr_i_fold + 1}/{self.h.num_folds}"
        self.log.info(self._sep(text))

    def end_fold(self):
        """Log fold completion, timing and metrics."""
        elapsed, elapsed_fold, remaining = self._time_stats()
        self.log.info(
            f"Fold {self.h.curr_i_fold}/{self.h.num_folds} completed in {elapsed_fold:.2f}s | "
            f"Total: {self._format_time(elapsed)} | Remaining: {self._format_time(remaining)}"
        )
        self.log.info(self._sep(f"Summary Fold {self.h.curr_i_fold}", width=60, sep='-'))
        self.log.info(
            f"{'Task':<10} | {'Best Epoch':<10} | {'Accuracy':<9} | {'F1 Score':<9} |"
        )
        for t in self.h.tasks:
            res = self.h.folds[self.h.curr_i_fold - 1].to(t)
            acc = res.metrics().val_accuracy[res.best_epoch]
            f1 = res.metrics().val_f1[res.best_epoch]
            self.log.info(
                f"{t:<10} | {res.best_epoch:^10d} | {acc * 100:>7.2f}%  | {f1 * 100:>7.2f}%  |"
            )

        for t in self.h.tasks:
            report = self.h.folds[self.h.curr_i_fold - 1].to(t).outcome().report
            self.log.info(self._sep(f"Report for {t}", width=60, sep='-'))
            self.display_report(report)

        self.log.info(self._sep("End of Fold", width=60))

    def display_report(self, report: dict):
        """Format and log a classification report."""
        self.log.info(
            f"{'Category':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}"
        )
        from configs.globals import CATEGORIES_MAP

        # Detailed per-class metrics
        for class_idx, metrics in report.items():
            if class_idx in ('macro avg', 'accuracy', 'weighted avg'):
                continue
            category = CATEGORIES_MAP.get(int(class_idx), f"Class {class_idx}")
            prec_str = self._format_metric(metrics.get('precision', ''), is_percentage=True)
            rec_str = self._format_metric(metrics.get('recall', ''), is_percentage=True)
            f1_str = self._format_metric(metrics.get('f1-score', ''), is_percentage=True)
            support = metrics.get('support', '')
            support_str = f"{support:>10}"
            self.log.info(f"{category:<12} | {prec_str} | {rec_str} | {f1_str} | {support_str}")

        # Macro average
        if 'macro avg' in report:
            macro = report['macro avg']
            p_str = self._format_metric(macro.get('precision', ''), is_percentage=True)
            r_str = self._format_metric(macro.get('recall', ''), is_percentage=True)
            f_str = self._format_metric(macro.get('f1-score', ''), is_percentage=True)
            self.log.info(
                f"{'macro avg':<12} | {p_str} | {r_str} | {f_str} | {'':>10}"
            )

    def end_training(self):
        """Log end of all folds."""
        self.log.info(self._sep("Training Complete"))
        self.log.info(self._sep("Summary", width=60, sep='-'))
        self.log.info(
            f"Model: {self.h.model_name} | "
            f"Folds: {self.h.num_folds} | "
            f"End at: {self.h.end_date}"
        )
        self.log.info(
            f"Tr. Time: {self._format_time(self.h.timer.get_duration())} | "
            f"Lrn. Rate: {self.h.model_parms.learning_rate} | "
            f"N. Epochs: {self.h.model_parms.num_epochs} | "
            f"Batch Size: {self.h.model_parms.batch_size}"
        )

        for task in self.h.tasks:
            self.log.info(self._sep(f"Avg. Task: {task}", width=60, sep='-'))

            # Aggregate metrics across folds
            aggregated = {}
            for fold in self.h.folds:
                report = fold.to(task).outcome().report
                for cls, metrics in report.items():
                    if cls not in aggregated:
                        aggregated[cls] = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
                    if cls not in ('accuracy', 'weighted avg'):
                        aggregated[cls]['precision'].append(metrics['precision'])
                        aggregated[cls]['recall'].append(metrics['recall'])
                        aggregated[cls]['f1-score'].append(metrics['f1-score'])
                        if 'support' in metrics:
                            aggregated[cls]['support'].append(metrics['support'])

            # Compute means and stds
            final_report = {}
            for cls, arrs in aggregated.items():
                if cls in ('accuracy', 'weighted avg'):
                    continue
                final_report[cls] = {
                    'precision': f"{(mean(arrs['precision'])*100):.2f} ± {(stdev(arrs['precision'])*100):.2f}",
                    'recall': f"{(mean(arrs['recall'])*100):.2f} ± {(stdev(arrs['recall'])*100):.2f}",
                    'f1-score': f"{(mean(arrs['f1-score'])*100):.2f} ± {(stdev(arrs['f1-score'])*100):.2f}",
                    'support': sum(arrs['support']),
                }

            self.display_report(final_report)

        self.log.info(self._sep("End of all folds", width=60))

    def flops(self):
        """Log FLOPS if available, else warn."""
        if self.h.flops:
            self.log.info(f"FLOPS: {self.h.flops.flops} | Params: {self.h.flops.params}")
        else:
            self.log.warning("FLOPS not calculated.")
