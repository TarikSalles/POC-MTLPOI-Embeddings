from abc import ABCMeta
from typing import Set


class HyperParams(metaclass=ABCMeta):
    """
    A base class for hyperparameters.
    """


class NeuralParams(HyperParams):
    """
    A class to keep track of the hyperparameters of a neural network.
    """

    def __init__(self, batch_size: int = 0, num_epochs: int = 0,
                 learning_rate: float = 0.0, optimizer: str = "",
                 optimizer_state: dict[str, object] = None, scheduler: str = "",
                 scheduler_state: dict[str, object] = None, criterion: dict[str, str] = None,
                 criterion_state: dict[str, dict[str, object]] = None, **kwargs):
        """
        Initialize the hyperparameters for a neural network.
        :param batch_size:
        :param num_epochs:
        :param learning_rate:
        :param optimizer:
        :param scheduler:
        :param criterion:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs
        self.learning_rate: float = learning_rate
        self.optimizer: str = optimizer
        self.optimizer_state: dict[str, str] = optimizer_state if optimizer_state else {}
        self.scheduler: str = scheduler
        self.scheduler_state: dict[str, str] = scheduler_state if scheduler_state else {}
        self.criterion: dict[str, str] = criterion if criterion else {}
        self.criterion_state: dict[str, dict[str, str]] = criterion_state if criterion_state else {}
