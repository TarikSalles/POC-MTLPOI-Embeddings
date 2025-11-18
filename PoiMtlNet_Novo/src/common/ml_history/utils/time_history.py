import time


class TimeHistory:
    """
    A class to keep track of the time taken for each fold in machine learning.
    """

    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.duration = None

    def start(self):
        """
        Start the timer.
        """
        self.start_time = time.time()

    def stop(self):
        """
        Stop the timer and calculate the duration.
        """
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def timer(self):
        """
        Get the current time.
        :return: The current time in seconds
        """
        return time.time() - self.start_time

    def get_duration(self):
        """
        Get the duration of the last recorded time.
        :return: The duration in seconds
        """
        if self.duration is None:
            raise ValueError("Timer has not been stopped yet.")
        return self.duration