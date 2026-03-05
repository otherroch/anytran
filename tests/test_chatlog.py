"""Tests for ChatLogger in anytran.chatlog."""
import os
import tempfile
import threading
import time
import unittest


class TestChatLogger(unittest.TestCase):

    def setUp(self):
        from anytran.chatlog import ChatLogger
        self.tmpdir = tempfile.mkdtemp()
        self.logger = ChatLogger(log_dir=self.tmpdir)

    def tearDown(self):
        self.logger.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_log_creates_file(self):
        from anytran.chatlog import ChatLogger
        self.logger.log("192.168.1.1", "Hello world")
        files = os.listdir(self.tmpdir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".txt"))

    def test_log_writes_content(self):
        self.logger.log("10.0.0.1", "Test message")
        files = os.listdir(self.tmpdir)
        self.assertEqual(len(files), 1)
        path = os.path.join(self.tmpdir, files[0])
        content = open(path, encoding="utf-8").read()
        self.assertIn("10.0.0.1", content)
        self.assertIn("Test message", content)

    def test_log_empty_text_no_file(self):
        self.logger.log("1.2.3.4", "")
        files = os.listdir(self.tmpdir)
        self.assertEqual(len(files), 0)

    def test_log_none_text_no_file(self):
        self.logger.log("1.2.3.4", None)
        files = os.listdir(self.tmpdir)
        self.assertEqual(len(files), 0)

    def test_log_strips_text(self):
        self.logger.log("1.2.3.4", "  spaced  ")
        files = os.listdir(self.tmpdir)
        path = os.path.join(self.tmpdir, files[0])
        content = open(path, encoding="utf-8").read()
        self.assertIn("spaced", content)

    def test_close_closes_file_handle(self):
        self.logger.log("1.2.3.4", "Before close")
        self.logger.close()
        self.assertIsNone(self.logger.file_handle)

    def test_close_idempotent(self):
        self.logger.log("1.2.3.4", "msg")
        self.logger.close()
        self.logger.close()  # Should not raise

    def test_multiple_logs_same_file(self):
        self.logger.log("1.1.1.1", "First")
        self.logger.log("2.2.2.2", "Second")
        files = os.listdir(self.tmpdir)
        self.assertEqual(len(files), 1)
        path = os.path.join(self.tmpdir, files[0])
        content = open(path, encoding="utf-8").read()
        self.assertIn("First", content)
        self.assertIn("Second", content)

    def test_log_format_has_timestamp(self):
        self.logger.log("1.2.3.4", "check_format")
        files = os.listdir(self.tmpdir)
        path = os.path.join(self.tmpdir, files[0])
        content = open(path, encoding="utf-8").read()
        # Timestamp format: YYYY-MM-DD HH:MM:SS
        import re
        self.assertRegex(content, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_get_log_filename_format(self):
        filename = self.logger._get_log_filename()
        basename = os.path.basename(filename)
        # Should match pattern like 20241201-1200.txt
        import re
        self.assertRegex(basename, r"\d{8}-\d{4}\.txt")

    def test_check_rotation_first_call(self):
        # First call sets current_hour
        result = self.logger._check_rotation()
        self.assertTrue(result)
        self.assertIsNotNone(self.logger.current_hour)

    def test_check_rotation_same_hour_no_rotate(self):
        self.logger._check_rotation()
        result = self.logger._check_rotation()
        self.assertFalse(result)

    def test_thread_safety(self):
        errors = []

        def log_messages():
            try:
                for i in range(10):
                    self.logger.log("1.2.3.4", f"thread message {i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=log_messages) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_extract_ip_from_rtsp_url_exported(self):
        from anytran.chatlog import extract_ip_from_rtsp_url
        ip = extract_ip_from_rtsp_url("rtsp://192.168.1.100:554/stream")
        self.assertEqual(ip, "192.168.1.100")


if __name__ == "__main__":
    unittest.main()
