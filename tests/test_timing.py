import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from anytran.timing import TimingsAggregator


class TestTimingsAggregator(unittest.TestCase):
    def test_format_stage_summary_empty(self):
        """Test format_stage_summary with no data"""
        agg = TimingsAggregator("test")
        result = agg.format_stage_summary()
        self.assertEqual(result, "")

    def test_format_stage_summary_stage1_only(self):
        """Test format_stage_summary with only Stage 1 data"""
        agg = TimingsAggregator("test")
        agg.add([("stage1_transcription", 2.0)], prefix="chunk")
        result = agg.format_stage_summary()
        self.assertIn("Total: 2.000s", result)
        self.assertIn("Stage 1 (Transcription): 2.000s (100.0%)", result)
        self.assertIn("Stage 2 (Translation):   0.000s (0.0%)", result)
        self.assertIn("Stage 3 (TTS):           0.000s (0.0%)", result)

    def test_format_stage_summary_all_stages(self):
        """Test format_stage_summary with all stages"""
        agg = TimingsAggregator("test")
        agg.add([
            ("stage1_transcription", 1.0),
            ("stage2_translation", 0.5),
            ("stage3_tts_synthesis", 0.3),
            ("stage3_tts_playback", 0.2),
        ], prefix="chunk")
        
        result = agg.format_stage_summary()
        
        # Total should be 2.0s
        self.assertIn("Total: 2.000s", result)
        
        # Stage 1: 1.0s = 50%
        self.assertIn("Stage 1 (Transcription): 1.000s (50.0%)", result)
        
        # Stage 2: 0.5s = 25%
        self.assertIn("Stage 2 (Translation):   0.500s (25.0%)", result)
        
        # Stage 3: 0.3 + 0.2 = 0.5s = 25%
        self.assertIn("Stage 3 (TTS):           0.500s (25.0%)", result)

    def test_format_stage_summary_multiple_chunks(self):
        """Test format_stage_summary with multiple chunks aggregated"""
        agg = TimingsAggregator("test")
        
        # Add first chunk
        agg.add([
            ("stage1_transcription", 1.0),
            ("stage2_translation", 0.5),
        ], prefix="chunk")
        
        # Add second chunk
        agg.add([
            ("stage1_transcription", 1.0),
            ("stage2_translation", 0.5),
        ], prefix="chunk")
        
        result = agg.format_stage_summary()
        
        # Total should be 3.0s (2.0s per chunk × 2 chunks)
        self.assertIn("Total: 3.000s", result)
        
        # Stage 1: 2.0s total
        self.assertIn("Stage 1 (Transcription): 2.000s", result)
        
        # Stage 2: 1.0s total
        self.assertIn("Stage 2 (Translation):   1.000s", result)

    def test_format_stage_summary_no_stage2(self):
        """Test format_stage_summary when Stage 2 (translation) is skipped"""
        agg = TimingsAggregator("test")
        agg.add([
            ("stage1_transcription", 3.0),
            ("stage3_tts_synthesis", 1.0),
        ], prefix="chunk")
        
        result = agg.format_stage_summary()
        
        # Total should be 4.0s
        self.assertIn("Total: 4.000s", result)
        
        # Stage 1: 3.0s = 75%
        self.assertIn("Stage 1 (Transcription): 3.000s (75.0%)", result)
        
        # Stage 2: 0.0s = 0%
        self.assertIn("Stage 2 (Translation):   0.000s (0.0%)", result)
        
        # Stage 3: 1.0s = 25%
        self.assertIn("Stage 3 (TTS):           1.000s (25.0%)", result)


if __name__ == "__main__":
    unittest.main()
