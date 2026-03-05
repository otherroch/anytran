"""Tests for anytran.stream_youtube utility functions."""
import unittest


class TestExtractYoutubeVideoId(unittest.TestCase):
    """Test the pure URL-parsing function extract_youtube_video_id."""

    def setUp(self):
        from anytran.stream_youtube import extract_youtube_video_id
        self.fn = extract_youtube_video_id

    def test_watch_url(self):
        result = self.fn("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(result, "dQw4w9WgXcQ")

    def test_short_url(self):
        result = self.fn("https://youtu.be/dQw4w9WgXcQ")
        self.assertEqual(result, "dQw4w9WgXcQ")

    def test_shorts_url(self):
        result = self.fn("https://www.youtube.com/shorts/dQw4w9WgXcQ")
        self.assertEqual(result, "dQw4w9WgXcQ")

    def test_non_youtube_url_returns_none(self):
        result = self.fn("https://example.com/video")
        self.assertIsNone(result)

    def test_empty_string(self):
        result = self.fn("")
        self.assertIsNone(result)

    def test_watch_url_with_extra_params(self):
        result = self.fn("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s")
        self.assertEqual(result, "dQw4w9WgXcQ")


class TestParseIso8601Duration(unittest.TestCase):
    """Test the parse_iso8601_duration function."""

    def setUp(self):
        from anytran.stream_youtube import parse_iso8601_duration
        self.fn = parse_iso8601_duration

    def test_hours_minutes_seconds(self):
        result = self.fn("PT1H30M45S")
        self.assertEqual(result, 1 * 3600 + 30 * 60 + 45)

    def test_minutes_only(self):
        result = self.fn("PT5M30S")
        self.assertEqual(result, 5 * 60 + 30)

    def test_seconds_only(self):
        result = self.fn("PT30S")
        self.assertEqual(result, 30)

    def test_empty_returns_none(self):
        result = self.fn("")
        self.assertIsNone(result)

    def test_none_returns_none(self):
        result = self.fn(None)
        self.assertIsNone(result)

    def test_hours_only(self):
        result = self.fn("PT2H")
        self.assertEqual(result, 7200)


if __name__ == "__main__":
    unittest.main()
