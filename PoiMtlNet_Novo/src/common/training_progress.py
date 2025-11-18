import time
from datetime import timedelta

from tqdm import tqdm


class TrainingProgressBar(tqdm):
    """Extended tqdm progress bar with specialized training tracking capabilities."""

    def __init__(self, num_epochs, dataloaders, **kwargs):
        """Initialize the training progress bar.

        Args:
            num_epochs: Total number of epochs
            dataloaders: List of dataloaders for training and validation
            category_dataloader: Dataloader for category prediction
            **kwargs: Additional arguments to pass to tqdm
        """
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders
        self.dataset_length = min(len(x) for x in dataloaders)
        total_steps = self.dataset_length * num_epochs

        # Initialize timers
        self.total_start_time = time.time()
        self.epoch_start_time = None

        self.current_epoch = 0

        # Initialize postfix storage
        self.metrics = {}

        # Initialize tqdm with unified progress bar
        super().__init__(total=total_steps, unit="batch",
                         desc=f"Epoch 1/{num_epochs}",
                         **kwargs)

    def __iter__(self):
        """Make the progress bar iterable over epochs.

        Yields:
            EpochContext: Context manager for each epoch
        """
        for epoch_idx in range(self.num_epochs):
            self.epoch_start_time = time.time()
            self.current_epoch = epoch_idx
            self.set_description(
                f"Epoch {self.current_epoch + 1}/{self.num_epochs}"
            )
            yield epoch_idx
            self.write("")

    def iter_epoch(self):
        """Iterate over the current epoch.

        Yields:
            EpochContext: Context manager for the current epoch
        """
        for data in len(self.dataloaders) == 1 and self.dataloaders or zip(*self.dataloaders):
            yield data
            self.update(1)

    def validation(self):
        """Context manager for tracking validation phase.

        Returns:
            Context manager for the validation phase
        """
        return self._ValidationContext(self)

    def set_postfix(self, dict_postfix=None, **kwargs):
        """Override tqdm's set_postfix to properly update our metrics dictionary.

        Args:
            dict_postfix: Dictionary of metrics to update
            **kwargs: Additional keyword arguments to update metrics
        """
        # Update our internal metrics dictionary
        if dict_postfix is not None:
            self.metrics.update(dict_postfix)
        if kwargs:
            self.metrics.update(kwargs)

        # Call the parent class method with our updated metrics
        super().set_postfix(**self.metrics)

        return self

    def close(self):
        """Override tqdm close method."""
        # Call parent class close method
        super().close()

    class _ValidationContext:
        """Context manager for validation phase."""

        def __init__(self, progress_bar):
            """Initialize the validation context."""
            self.progress_bar = progress_bar

        def __enter__(self):
            """Enter the validation context."""

            self.progress_bar.set_description(
                f"Epoch {self.progress_bar.current_epoch + 1}/{self.progress_bar.num_epochs} [Validating]"
            )
            return self.progress_bar

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Exit the validation context."""
            if exc_type is None:
                # Calculate total epoch time for description
                epoch_time = time.time() - self.progress_bar.epoch_start_time

                # Update description with completion info
                self.progress_bar.set_description(
                    f"Epoch {self.progress_bar.current_epoch + 1}/{self.progress_bar.num_epochs} "
                    f"[{timedelta(seconds=int(epoch_time))}]"
                )

                # Update the progress bar
                self.progress_bar.set_postfix(**self.progress_bar.metrics)
            return False
