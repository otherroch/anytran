"""Tests for anytran.timing module."""
import time
import unittest


class TestAddTiming(unittest.TestCase):
    def test_none_timings_returns_early(self):
        from anytran.timing import add_timing
        add_timing(None, "label", time.perf_counter())  # Should not raise

    def test_adds_to_list(self):
        from anytran.timing import add_timing
        timings = []
        start = time.perf_counter()
        time.sleep(0.001)
        add_timing(timings, "test_label", start)
        self.assertEqual(len(timings), 1)
        label, elapsed = timings[0]
        self.assertEqual(label, "test_label")
        self.assertGreater(elapsed, 0)

    def test_multiple_adds(self):
        from anytran.timing import add_timing
        timings = []
        for i in range(3):
            add_timing(timings, f"step{i}", time.perf_counter())
        self.assertEqual(len(timings), 3)


class TestFormatTiming(unittest.TestCase):
    def test_empty_returns_empty_string(self):
        from anytran.timing import format_timing
        self.assertEqual(format_timing([]), "")

    def test_none_returns_empty_string(self):
        from anytran.timing import format_timing
        self.assertEqual(format_timing(None), "")

    def test_formats_single_entry(self):
        from anytran.timing import format_timing
        result = format_timing([("step1", 0.123)])
        self.assertIn("step1", result)
        self.assertIn("ms", result)

    def test_formats_multiple_entries(self):
        from anytran.timing import format_timing
        result = format_timing([("a", 0.1), ("b", 0.2)])
        self.assertIn("a", result)
        self.assertIn("b", result)


class TestTimingsAggregator(unittest.TestCase):
    def test_initial_format_summary_empty(self):
        from anytran.timing import TimingsAggregator
        agg = TimingsAggregator("test")
        self.assertEqual(agg.format_summary(), "")

    def test_add_time(self):
        from anytran.timing import TimingsAggregator
        agg = TimingsAggregator()
        now = time.perf_counter()
        agg.add_time("step1", now, now + 0.1)
        summary = agg.format_summary()
        self.assertIn("step1", summary)

    def test_add_multiple_times_same_label(self):
        from anytran.timing import TimingsAggregator
        agg = TimingsAggregator()
        now = time.perf_counter()
        agg.add_time("step1", now, now + 0.1)
        agg.add_time("step1", now, now + 0.2)
        summary = agg.format_summary()
        self.assertIn("step1", summary)

    def test_add_timings_list(self):
        from anytran.timing import TimingsAggregator
        agg = TimingsAggregator()
        timings = [("a", 0.1), ("b", 0.2)]
        agg.add(timings)
        summary = agg.format_summary()
        self.assertIn("a", summary)
        self.assertIn("b", summary)

    def test_add_timings_with_prefix(self):
        from anytran.timing import TimingsAggregator
        agg = TimingsAggregator()
        timings = [("step1", 0.1)]
        agg.add(timings, prefix="backend")
        summary = agg.format_summary()
        self.assertIn("backend.step1", summary)

    def test_add_empty_timings_returns_early(self):
        from anytran.timing import TimingsAggregator
        agg = TimingsAggregator()
        agg.add([])  # Should not raise or add entries
        self.assertEqual(agg.format_summary(), "")

    def test_add_none_returns_early(self):
        from anytran.timing import TimingsAggregator
        agg = TimingsAggregator()
        agg.add(None)  # Should not raise
        self.assertEqual(agg.format_summary(), "")

    def test_format_translate_overhead_empty(self):
        from anytran.timing import TimingsAggregator
        agg = TimingsAggregator()
        result = agg.format_translate_overhead("whispercpp")
        self.assertEqual(result, "")

    def test_format_translate_overhead_with_data(self):
        from anytran.timing import TimingsAggregator
        agg = TimingsAggregator()
        now = time.perf_counter()
        agg.add_time("chunk.translate_total", now, now + 0.5)
        agg.add_time("whispercpp.transcribe", now, now + 0.3)
        result = agg.format_translate_overhead("whispercpp")
        self.assertIn("overhead", result)

    def test_format_stage_summary_empty(self):
        from anytran.timing import TimingsAggregator
        agg = TimingsAggregator()
        result = agg.format_stage_summary()
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
