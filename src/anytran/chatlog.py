from datetime import datetime
import os
import threading

from .utils import extract_ip_from_rtsp_url


class ChatLogger:
    def __init__(self, log_dir="."):
        self.log_dir = log_dir
        self.current_file = None
        self.current_hour = None
        self.file_handle = None
        self.lock = threading.Lock()  # Add thread safety

    def _get_log_filename(self):
        now = datetime.now()
        return os.path.join(self.log_dir, f"{now.strftime('%Y%m%d-%H00')}.txt")

    def _check_rotation(self):
        current_hour = datetime.now().strftime("%Y%m%d-%H")
        if current_hour != self.current_hour:
            if self.file_handle:
                self.file_handle.close()
            self.current_hour = current_hour
            self.current_file = self._get_log_filename()
            self.file_handle = open(self.current_file, "a", encoding="utf-8")
            return True
        return False

    def log(self, rtsp_ip, text):
        if not text:
            return

        with self.lock:  # Thread-safe file operations
            self._check_rotation()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp},{rtsp_ip},{text.strip()}\n"

            if self.file_handle:
                self.file_handle.write(log_entry)
                self.file_handle.flush()

    def close(self):
        with self.lock:
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None


__all__ = ["ChatLogger", "extract_ip_from_rtsp_url"]
