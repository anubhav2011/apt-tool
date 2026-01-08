import logging
from datetime import datetime, timezone, timedelta
import os
from types import TracebackType
from typing import Mapping
from pathlib import Path

from dotenv import load_dotenv
import glob

load_dotenv()
ENV = os.getenv('APP_ENV')


class DebugLogger:
    def __init__(self, logger_name="debug_logger"):
        self._logger_name = logger_name
        # Calculate paths directly here instead
        self._root_dir = Path(__file__).resolve().parents[2]
        self._log_dir = "debug_logs"
        self._logs_dir = str(self._root_dir / self._log_dir)
        self._debug_logger: logging.Logger = self._create_logger()
        self._clean_old_logs()

    def _create_logger(self):
        """Creates and returns a logger with date-wise log files."""
        logger = logging.getLogger(self._logger_name)
        logger.setLevel(logging.DEBUG)

        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_filename = str(Path(self._logs_dir) / f"debug_logs_{current_date}.log")

        # Ensure the log directory exists
        if not os.path.exists(self._logs_dir):
            os.makedirs(self._logs_dir)

        # Clear existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Define log file and console handlers with UTF-8 encoding
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        console_handler = logging.StreamHandler()

        # Force UTF-8 encoding for console handler on Windows
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8')
            except Exception:
                pass  # If reconfigure fails, continue without it

        # Set log levels - file handler gets all logs, console only gets INFO
        file_handler.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.INFO)

        # Define log format - remove level name from console output
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_format = logging.Formatter('%(message)s')  # Simplified format for console

        # Attach formatters to handlers
        file_handler.setFormatter(file_format)
        console_handler.setFormatter(console_format)

        # Attach handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False

        return logger

    def _clean_old_logs(self, days_to_keep=30):
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            # Find all .log files in the directory
            log_files = glob.glob(os.path.join(self._logs_dir, "*.log"))

            deleted_files = []
            for log_file in log_files:
                # Get file modification time
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))

                if file_mod_time < cutoff_date:
                    os.remove(log_file)
                    deleted_files.append(log_file)
                    print(f"Deleted: {log_file}")

            if deleted_files:
                print(f"Cleanup completed: Deleted {len(deleted_files)} old log files")
            else:
                print("No old log files found to delete")

            return deleted_files

        except Exception as e:
            print(f"Error cleaning up logs: {e}")
            return []

    def info(self,
             msg: object,
             *args: object,
             exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                 None, None, None] | BaseException = None,
             stack_info: bool = False,
             stacklevel: int = 1,
             extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.info(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                stacklevel=stacklevel)

    def debug(self,
              msg: object,
              *args: object,
              exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                  None, None, None] | BaseException = None,
              stack_info: bool = False,
              stacklevel: int = 1,
              extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.debug(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                 stacklevel=stacklevel)

    def warning(self,
                msg: object,
                *args: object,
                exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                    None, None, None] | BaseException = None,
                stack_info: bool = False,
                stacklevel: int = 1,
                extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.warning(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                   stacklevel=stacklevel)

    def error(self,
              msg: object,
              *args: object,
              exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                  None, None, None] | BaseException = None,
              stack_info: bool = False,
              stacklevel: int = 1,
              extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.error(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                 stacklevel=stacklevel)

    def critical(self,
                 msg: object,
                 *args: object,
                 exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                     None, None, None] | BaseException = None,
                 stack_info: bool = False,
                 stacklevel: int = 1,
                 extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.critical(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                    stacklevel=stacklevel)

    def exception(self,
                  msg: object,
                  *args: object,
                  exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                      None, None, None] | BaseException = True,
                  stack_info: bool = False,
                  stacklevel: int = 1,
                  extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.exception(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                     stacklevel=stacklevel)


debug_logger = DebugLogger()
