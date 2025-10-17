from __future__ import annotations

import time
from copy import deepcopy
from typing import List, Set, Optional

from utils.ml_history.display import HistoryDisplay
from utils.ml_history.parms.neural import HyperParams
from utils.ml_history.storage import HistoryStorage
from utils.ml_history.utils.dataset import DatasetHistory
from utils.ml_history.utils.time_history import TimeHistory


class TaskTrainMetric:
    """
    A class to keep track of raw metrics for a machine learning model.
    """

    def __init__(self):
        self.loss: List[float] = []
        self.accuracy: List[float] = []
        self.f1: List[float] = []

        self.val_loss: List[float] = []
        self.val_accuracy: List[float] = []
        self.val_f1: List[float] = []

    def add(self, loss: float, accuracy: float, f1: float = 0.0):
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.f1.append(f1)

    def add_val(self, val_loss: float, val_accuracy: float, val_f1: float = 0.0):
        self.val_loss.append(val_loss)
        self.val_accuracy.append(val_accuracy)
        self.val_f1.append(val_f1)


class TaskOutcome:
    """
    A class to keep track of the final outcome of a machine learning model.
    """

    def __init__(self):
        self.report: dict = {}


class FoldHistoryMetric:
    """
    A class to keep track of the history of multiple folds in machine learning.
    """

    def __init__(self):
        self.task_metrics: TaskTrainMetric = TaskTrainMetric()
        self.task_outcome: TaskOutcome = TaskOutcome()
        self.best_model: dict = {}
        self.best_epoch: int = 0
        self.best_time: float = 0.0

    def metrics(self):
        """
        Set the metrics for a specific task.
        :param task: The task name
        :param metric: The TaskTrainMetric object
        """
        return self.task_metrics

    def outcome(self):
        """
        Set the outcome for a specific task.
        :param task: The task name
        :param outcome: The TaskOutcome object
        """
        return self.task_outcome

    def add(self, loss: float, accuracy: float, f1: float = 0.0):
        """
        Add metrics for a specific task.
        :param task: The task name
        :param loss: The loss value
        :param accuracy: The accuracy value
        :param precision: The precision value
        :param recall: The recall value
        :param f1: The F1 score value
        """
        self.task_metrics.add(loss, accuracy, f1)

    def add_val(self, val_loss: float, val_accuracy: float, val_f1: float = 0.0,
                model_state: Optional[dict] = None, best_metric: str = 'val_f1',
                best_time: Optional[float] = None):
        """
        Add validation metrics for a specific task.
        :param best_time:
        :param val_loss: The validation loss value
        :param val_accuracy: The validation accuracy value
        :param val_f1: The validation F1 score value
        :param model_state: The model state dictionary
        :param best_metric: Metric to use for determining best model ('val_loss', 'val_accuracy', 'val_f1')
        """
        self.task_metrics.add_val(val_loss, val_accuracy, val_f1)

        if not model_state:
            return

        metric_value = None
        prev_best = None
        comparison = lambda x, y: x >= y
        if best_metric == 'val_f1':
            metric_value = val_f1
            prev_best = max(self.task_metrics.val_f1, default=0)
        elif best_metric == 'val_accuracy':
            metric_value = val_accuracy
            prev_best = max(self.task_metrics.val_accuracy, default=0)
        elif best_metric == 'val_loss':
            metric_value = val_loss
            comparison = lambda x, y: x <= y
            prev_best = min(self.task_metrics.val_loss, default=float('inf'))

        if comparison(metric_value, prev_best):
            self.best_model = deepcopy(model_state)
            self.best_epoch = len(self.task_metrics.val_f1) - 1
            self.best_time = best_time

    def add_report(self, report: dict):
        """
        Add a report for a specific task.
        :param task: The task name
        :param report: The report dictionary
        """
        self.task_outcome.report = report


class FoldHistory:
    """
    A class to keep track of the history of a single fold in machine learning.
    """

    def __init__(self, fold_number, tasks: Set[str]):
        self.timer: TimeHistory = TimeHistory()
        self.fold_number = fold_number
        self.model: FoldHistoryMetric = FoldHistoryMetric()
        self.tasks_history = {
            task: FoldHistoryMetric() for task in tasks
        }

        self.extra_history = {}

    @classmethod
    def standalone(cls, tasks: Set[str]):
        """
        Create a standalone FoldHistory object.
        :param tasks: The tasks to be tracked
        :return: A new FoldHistory object
        """
        history = cls(fold_number=0, tasks=tasks)
        history.start()
        return history

    def to(self, task: str):
        """
        Set the metrics for a specific task.
        :param task: The task name
        :param metric: The TaskTrainMetric object
        """
        if task not in self.tasks_history:
            raise ValueError(f"Task {task} not found in fold {self.fold_number}.")

        return self.tasks_history[task]

    def start(self):
        self.timer.start()

    def end(self):
        self.timer.stop()


class FlopsMetrics:
    """Class to hold flops metrics."""

    def __init__(self, flops, params):
        self.flops: List[float] = flops
        self.params: List[float] = params
        self.memory: List[float] = []
        self.inference_time: List[float] = []
        self.training_time: List[float] = []

    def to_dict(self) -> dict:
        return {
            'flops': self.flops,
            'params': self.params,
            'memory': self.memory,
            'inference_time': self.inference_time,
            'training_time': self.training_time
        }


class _MLHistoryContext:
    """
    A context manager for MLHistory.
    """

    def __init__(self, ml_history: MLHistory):
        self.ml_history = ml_history

    def __enter__(self):
        self.ml_history.start()
        return self.ml_history

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ml_history.end()


class _MLHistoryIterator:
    """
    An iterator for MLHistory.
    """

    def __init__(self, ml_history: MLHistory):
        self.ml_history = ml_history

    def __iter__(self):
        self.ml_history.start()
        for i in range(len(self.ml_history.folds)):
            yield self.ml_history.folds[i]
        self.ml_history.end()
        return self

    def __next__(self):
        if self.ml_history.curr_i_fold >= len(self.ml_history.folds):
            raise StopIteration
        fold = self.ml_history.folds[self.ml_history.curr_i_fold]
        self.ml_history.curr_i_fold += 1
        return fold


class MLHistory:
    """
    A class to keep track of the history of machine learning model training.
    """

    def __init__(self,
                 model_name: str,
                 tasks: Set[str] | str,
                 model_type: str = 'MTL',
                 num_folds: int = 1,
                 model_parms: Optional[HyperParams] = None,
                 datasets: Optional[Set[DatasetHistory]] = None):
        """
        Initialize the MLHistory object.
        :param model_name:
        :param num_folds:
        :param datasets:
        """

        # Model configuration
        self.model_name = model_name
        self.model_type = model_type
        self.num_folds = num_folds
        self.model_parms: HyperParams = model_parms
        self.model_arch: Optional[str] = None
        self.datasets: Optional[Set[DatasetHistory]] = datasets

        # Training configuration
        self.tasks = isinstance(tasks, str) and {tasks} or tasks
        self.folds: List[FoldHistory] = [
            FoldHistory(fold_number=i, tasks=self.tasks) for i in range(num_folds)
        ]
        self.flops: FlopsMetrics = None

        # Internal states
        self.timer: TimeHistory = TimeHistory()
        self.start_date = None
        self.end_date = None
        self.curr_i_fold = 0
        self.display = HistoryDisplay(self)
        self.storage = HistoryStorage(self)

    def context(self):
        """
        Context manager for MLHistory.
        :return: MLHistory object
        """
        return _MLHistoryContext(self)

    def iterator(self):
        """
        Iterator for MLHistory.
        :return: MLHistory object
        """
        return _MLHistoryIterator(self)

    def get_curr_fold(self) -> FoldHistory:
        """
        Get the current fold.
        :return: The current fold number
        """
        return self.folds[self.curr_i_fold]

    def set_model_parms(self, model_parms: HyperParams):
        """
        Set the model parameters.
        :param model_parms: HyperParams object
        """
        self.model_parms = model_parms

    def set_model_arch(self, model_arch: str):
        """
        Set the model architecture.
        :param model_arch: The model architecture
        """
        self.model_arch = model_arch

    def start(self):
        """
        Start the timer for the current fold.
        """
        self.timer.start()
        self.start_date = time.strftime("%Y%m%d_%H%M")
        self.get_curr_fold().start()

    def step(self):
        """
        Increment the current fold.
        """
        self.folds[self.curr_i_fold].end()
        if self.curr_i_fold >= self.num_folds - 1:
            self.end()
            return
        self.curr_i_fold += 1
        self.folds[self.curr_i_fold].timer.start()

    def set_flops(self, flops: FlopsMetrics):
        """
        Set the flops metrics.
        :param flops: FlopsMetrics object
        """
        self.flops = flops

    def end(self):
        """
        End the timer for the current fold.
        """
        self.timer.stop()
        self.end_date = time.strftime("%Y%m%d_%H%M%S")
