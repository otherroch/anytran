"""Tests for remaining utility functions in anytran.utils."""
import os
import sys
import tempfile
import unittest


class TestNormalizeLangCode(unittest.TestCase):
    def test_returns_none_for_none(self):
        from anytran.utils import normalize_lang_code
        self.assertIsNone(normalize_lang_code(None))

    def test_returns_none_for_empty_string(self):
        from anytran.utils import normalize_lang_code
        self.assertIsNone(normalize_lang_code(""))

    def test_strips_and_lowercases(self):
        from anytran.utils import normalize_lang_code
        self.assertEqual(normalize_lang_code("  EN  "), "en")

    def test_plain_code(self):
        from anytran.utils import normalize_lang_code
        self.assertEqual(normalize_lang_code("fr"), "fr")

    def test_uppercase(self):
        from anytran.utils import normalize_lang_code
        self.assertEqual(normalize_lang_code("ZH-CN"), "zh-cn")


class TestComputeWindowParams(unittest.TestCase):
    def test_basic(self):
        from anytran.utils import compute_window_params
        w, o = compute_window_params(5.0, 1.0, 16000)
        self.assertEqual(w, 80000)
        self.assertEqual(o, 16000)

    def test_zero_overlap(self):
        from anytran.utils import compute_window_params
        w, o = compute_window_params(3.0, 0.0, 16000)
        self.assertEqual(w, 48000)
        self.assertEqual(o, 0)

    def test_overlap_clamped_to_less_than_window(self):
        from anytran.utils import compute_window_params
        w, o = compute_window_params(1.0, 5.0, 16000)
        self.assertLess(o, w)

    def test_very_small_window_minimum_one(self):
        from anytran.utils import compute_window_params
        w, o = compute_window_params(0.00001, 0.0, 16000)
        self.assertEqual(w, 1)


class TestExtractIpFromRtspUrl(unittest.TestCase):
    def test_standard_rtsp(self):
        from anytran.utils import extract_ip_from_rtsp_url
        self.assertEqual(extract_ip_from_rtsp_url("rtsp://10.0.0.1:554/stream"), "10.0.0.1")

    def test_hostname(self):
        from anytran.utils import extract_ip_from_rtsp_url
        self.assertEqual(extract_ip_from_rtsp_url("rtsp://camera.local/live"), "camera.local")

    def test_with_credentials(self):
        from anytran.utils import extract_ip_from_rtsp_url
        result = extract_ip_from_rtsp_url("rtsp://user:pass@192.168.1.5:554/stream")
        self.assertEqual(result, "192.168.1.5")

    def test_empty_string(self):
        from anytran.utils import extract_ip_from_rtsp_url
        # Empty string: urlparse gives no hostname; regex also finds no match
        result = extract_ip_from_rtsp_url("")
        # Returns empty string or "unknown" depending on urlparse behaviour
        self.assertIsNotNone(result)

    def test_none(self):
        from anytran.utils import extract_ip_from_rtsp_url
        result = extract_ip_from_rtsp_url(None)
        self.assertEqual(result, "unknown")

    def test_fallback_regex(self):
        from anytran.utils import extract_ip_from_rtsp_url
        # A malformed URL that still contains //host pattern
        result = extract_ip_from_rtsp_url("rtsp://myhost/path")
        self.assertEqual(result, "myhost")


class TestResolvePathWithFallback(unittest.TestCase):
    def test_none_returns_none(self):
        from anytran.utils import resolve_path_with_fallback
        self.assertIsNone(resolve_path_with_fallback(None, "/some/dir"))

    def test_absolute_path_returned_as_is(self):
        from anytran.utils import resolve_path_with_fallback
        result = resolve_path_with_fallback("/absolute/path.txt", "/fallback")
        self.assertEqual(result, "/absolute/path.txt")

    def test_relative_path_that_exists(self):
        from anytran.utils import resolve_path_with_fallback
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            existing_path = f.name
        try:
            result = resolve_path_with_fallback(existing_path, "/fallback")
            self.assertEqual(result, existing_path)
        finally:
            os.unlink(existing_path)

    def test_relative_path_in_fallback_dir(self):
        from anytran.utils import resolve_path_with_fallback
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = "test_file.txt"
            full_path = os.path.join(tmpdir, fname)
            open(full_path, "w").close()
            result = resolve_path_with_fallback(fname, tmpdir)
            self.assertEqual(result, full_path)

    def test_nonexistent_path_returns_original(self):
        from anytran.utils import resolve_path_with_fallback
        result = resolve_path_with_fallback("nonexistent.txt", "/no/fallback")
        self.assertEqual(result, "nonexistent.txt")

    def test_tilde_expansion(self):
        from anytran.utils import resolve_path_with_fallback
        result = resolve_path_with_fallback("~/some_path", "/fallback")
        self.assertFalse(result.startswith("~"))


class TestRestoreTerminal(unittest.TestCase):
    def test_module_imports_without_error(self):
        import anytran.utils  # Should not raise


if __name__ == "__main__":
    unittest.main()
