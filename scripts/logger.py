import logging
import os

class SetupLogger:
    """
    A class to configure logging for the application.

    Attributes
    ----------
    log_file : str
        Path to the log file.
    log_level : int
        Logging level (e.g., logging.INFO).
    """

    def __init__(self, log_file='logs/app.log', log_level=logging.INFO):
        """
        Initialize the logger with the given log file and level.

        Parameters
        ----------
        log_file : str
            File path where logs will be saved.
        log_level : int
            Logging level (default: logging.INFO).
        """
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Prevent adding multiple handlers in environments like Jupyter or Flask reloading
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.propagate = False  # Optional: Avoid double logging

    def get_logger(self):
        """
        Get the configured logger instance.

        Returns
        -------
        logging.Logger
            The logger instance.
        """
        return self.logger
