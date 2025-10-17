import os
from pathlib import Path
import json
from typing import Any, Dict, List, Union, TYPE_CHECKING
from statistics import mean, stdev

import pandas as pd
import matplotlib.pyplot as plt

from configs.globals import CATEGORIES_MAP

if TYPE_CHECKING:
    from utils.ml_history.metrics import MLHistory


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_text(data: str, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        f.write(data)


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    df.to_csv(path, index=False, **kwargs)


def map_category(key: Union[int, float, str]) -> str:
    """Convert numeric or string keys to category names."""
    try:
        idx = int(float(key))
        return CATEGORIES_MAP.get(idx, f"Unknown-{key}")
    except Exception:
        return str(key)


class SummaryGenerator:
    """Generate overall and per-category performance summaries."""

    def __init__(self, history: Any):
        self.history = history

    def generate(self, out_dir: Path) -> None:
        out = ensure_dir(out_dir)
        perf = self._collect_performance()
        cat_metrics = self._collect_category_metrics()

        # Overall summary
        stats = {
            task: {
                metric: self._stats(vals)
                for metric, vals in metrics.items()
            }
            for task, metrics in perf.items()
        }
        save_json(stats, out / 'full_summary.json')

        # Category summaries
        for task, cats in cat_metrics.items():
            df = pd.DataFrame([
                {
                    'Category': cat,
                    **{m: mean(vals) if vals else 0 for m, vals in metrics.items()},
                }
                for cat, metrics in cats.items()])
            df_fmt = pd.DataFrame([{
                'Category': cat,
                **{m: f"{mean(vals) * 100:.2f} ± {stdev(vals) * 100:.2f}" for m, vals in metrics.items()},
            } for cat, metrics in cats.items()])

            if not df.empty:
                save_csv(df, out / f'summary_{task}_metrics.csv', float_format='%.4f')

            if not df_fmt.empty:
                save_csv(df_fmt, out / f'summary_{task}_metrics_formatted.csv', float_format='%.4f')

    @staticmethod
    def _stats(values: List[float]) -> Dict[str, float]:
        """Compute mean, std, min, max."""
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        return {
            'mean': mean(values),
            'std': stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
        }

    def _collect_performance(self) -> Dict[str, Dict[str, List[float]]]:
        perf: Dict[str, Dict[str, List[float]]] = {
            task: {'accuracy': [], 'f1': []}
            for task in self.history.tasks
        }
        for fold in self.history.folds:
            for task in self.history.tasks:
                th = fold.tasks_history.get(task)
                if not th:
                    continue
                best = th.best_epoch
                for metric in ('val_accuracy', 'val_f1'):
                    data = getattr(th.task_metrics, metric, [])
                    if best < len(data):
                        perf[task][metric.split('_')[1]].append(data[best])
        return perf

    def _collect_category_metrics(self) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        result: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for fold in self.history.folds:
            for task in self.history.tasks:
                th = fold.tasks_history.get(task)
                if not th or not getattr(th.task_outcome, 'report', None):
                    continue
                for cat, vals in th.task_outcome.report.items():
                    if not isinstance(vals, dict) or cat in ('accuracy', 'weighted avg'):
                        continue
                    name = cat if cat == 'macro avg' else map_category(cat)
                    task_dict = result.setdefault(task, {})
                    metrics = task_dict.setdefault(name, {'precision': [], 'recall': [], 'f1-score': [], 'support': []})
                    for m in metrics:
                        if m in vals:
                            metrics[m].append(vals[m])
        return result


class HistoryStorage:
    """Save model configuration, metrics, reports, summaries, and plots."""

    def __init__(self, history: 'MLHistory') -> None:
        self.history = history

    def save(self, path: Union[str, Path]) -> Path:
        base = ensure_dir(Path(path) / self._folder_name())
        dirs = {k: ensure_dir(base / k) for k in ('model', 'metrics', 'folds', 'summary', 'plots')}

        self._save_params(dirs['model'])
        self._save_metrics(dirs['metrics'])
        self._save_reports(dirs['folds'])
        SummaryGenerator(self.history).generate(dirs['summary'])
        self._save_plots(dirs['plots'])

        return base

    def _folder_name(self) -> str:
        p = self.history.model_parms
        parts = [self.history.model_name, f"lr{p.learning_rate:.1e}", f"bs{p.batch_size}", f"ep{p.num_epochs}",
                 self.history.start_date]
        return "_".join(str(x).lower() for x in parts)

    def _save_params(self, path: Path) -> None:
        params = {
            'model': {'name': self.history.model_name, 'type': self.history.model_type},
            'training': {
                'folds': self.history.num_folds,
                'tasks': list(self.history.tasks),
                'dates': {'start': self.history.start_date, 'end': self.history.end_date}
            },
            'datasets': [
                ds.to_json()
                for ds in self.history.datasets
            ],
            "flops": self.history.flops.to_dict() if self.history.flops is not None else None,
            'hyperparameters': {k: v for k, v in vars(self.history.model_parms).items() if not k.startswith('_')}
        }

        try:
            save_json(params, path / 'model_params.json')
        except Exception as e:
            print(f"Error saving model parameters: {e}")
            save_text(str(params), path / 'model_params.txt')

        if self.history.model_arch:
            with open(path / 'arch.txt', 'w') as f:
                f.write(self.history.model_arch)

    def _save_metrics(self, path: Path) -> None:
        for i, fold in enumerate(self.history.folds, start=1):
            for task in self.history.tasks:
                th = fold.tasks_history.get(task)
                if not th:
                    continue
                for phase in ('', 'val_'):
                    df = pd.DataFrame({
                        'epoch': list(range(1, len(getattr(th.task_metrics, phase + 'loss')) + 1)),
                        **{metric: getattr(th.task_metrics, phase + metric) for metric in ('loss', 'accuracy', 'f1')}
                    })
                    save_csv(df, path / f'fold{i}_{task}_{phase or "train"}.csv')

    def _save_reports(self, path: Path) -> None:
        for i, fold in enumerate(self.history.folds, start=1):
            fold_info: Dict[str, Any] = {
                'fold_number': i,
                'duration': fold.timer.get_duration() if fold.timer.duration else 0,
                'best_epochs': {}
            }
            for task in self.history.tasks:
                th = fold.tasks_history.get(task)
                if not th or not getattr(th.task_outcome, 'report', None):
                    continue
                report = {map_category(k): v for k, v in th.task_outcome.report.items() if isinstance(v, dict)}
                save_json(report, path / f'fold{i}_{task}_report.json')
                df = pd.DataFrame([{'Category': k, **v} for k, v in report.items()])

                # Best epoch
                be = th.best_epoch
                acc = th.task_metrics.val_accuracy[be] if be < len(th.task_metrics.val_accuracy) else None
                f1 = th.task_metrics.val_f1[be] if be < len(th.task_metrics.val_f1) else None
                fold_info['best_epochs'][task] = {'epoch': be, 'accuracy': acc, 'f1': f1, 'time': th.best_time}

                save_csv(df, path / f'fold{i}_{task}_report.csv', float_format='%.4f')
            save_json(fold_info, path / f'fold{i}_info.json')

    def _save_plots(self, path: Path) -> None:
        for metric in ('loss', 'accuracy', 'f1'):
            for task in self.history.tasks | {'model'}:
                plt.figure()
                for i, fold in enumerate(self.history.folds, start=1):
                    th = fold.tasks_history.get(task) if task != 'model' else getattr(fold, 'model', None)
                    if not th:
                        continue
                    tm = th.task_metrics
                    x = range(1, len(getattr(tm, metric)) + 1)
                    plt.plot(x, getattr(tm, metric), label=f'Fold{i} Train')
                    plt.plot(x, getattr(tm, f'val_{metric}'), linestyle='--', label=f'Fold{i} Val')
                plt.title(f'{metric.title()} over Epochs - {task}')
                plt.legend();
                plt.grid(True)
                save_path = ensure_dir(path / task) / f"{metric}.png"
                plt.savefig(save_path);
                plt.close()
