"""
Test to verify the slate-voice language bug fixes.

Tests two scenarios:
1. Langswap scenario where slate audio should use tts_lang (not hardcoded 'en')
2. Translation scenario where both scribe and slate should be generated
"""

import pytest
import numpy as np
from unittest import mock

from anytran import processing


@mock.patch('anytran.processing.synthesize_tts_pcm_with_cloning')
@mock.patch('anytran.processing.translate_text')
@mock.patch('anytran.processing.translate_audio')
def test_langswap_slate_uses_tts_lang(mock_translate_audio, mock_translate_text, mock_synthesize):
    """Test that langswap scenario uses tts_lang instead of hardcoded 'en'."""

    captured_calls = []

    def capture_synthesize(text, rate, output_lang, **kwargs):
        captured_calls.append({
            'text': text,
            'language': output_lang,
        })
        return np.zeros(1000, dtype=np.int16)

    mock_synthesize.side_effect = capture_synthesize
    mock_translate_audio.return_value = (np.zeros(16000, dtype=np.int16), "Hello world", "en")
    # Must return a translation so Stage 2 runs and sets stage2_ran = True
    mock_translate_text.return_value = "Bonjour le monde"

    audio_segment = (np.random.randn(16000) * 1000).astype(np.int16)
    slate_tts_segments = []

    # Use non-English text_translation_target so Stage 2 runs, which triggers
    # the slate TTS synthesis path (line: if stage2_ran and translated_text ...)
    from anytran.pipeline_config import PipelineConfig
    result = processing.process_audio_chunk(
        audio_segment,
        16000,
        PipelineConfig(
            input_lang="en",
            output_lang="en",
            magnitude_threshold=0.001,
            model="small",
            verbose=False,
            scribe_backend="auto",
            text_translation_target="fr",  # French target so Stage 2 runs
            slate_backend="googletrans",
            voice_backend="gtts",
            voice_model=None,
            voice_lang="fr",  # Override to French!
        ),
        stream_id=None,
        timing_stats=None,
        slate_tts_segments=slate_tts_segments,
    )
    
    # Should have generated slate audio in French (from voice_lang)
    assert len(captured_calls) == 1
    assert captured_calls[0]['language'] == 'fr', \
        f"Expected French (fr), got {captured_calls[0]['language']}"


@mock.patch('anytran.processing.synthesize_tts_pcm_with_cloning')
@mock.patch('anytran.processing.translate_text')
@mock.patch('anytran.processing.translate_audio')
def test_scribe_and_slate_both_generated_during_translation(mock_translate_audio, mock_translate_text, mock_synthesize):
    """Test that both scribe and slate are generated when translating."""

    captured_calls = []

    def capture_synthesize(text, rate, output_lang, **kwargs):
        captured_calls.append({
            'text': text,
            'language': output_lang,
        })
        return np.zeros(1000, dtype=np.int16)

    mock_synthesize.side_effect = capture_synthesize
    mock_translate_audio.return_value = (np.zeros(16000, dtype=np.int16), "Hello world", "en")
    mock_translate_text.return_value = "Bonjour le monde"

    audio_segment = (np.random.randn(16000) * 1000).astype(np.int16)
    scribe_tts_segments = []
    slate_tts_segments = []

    # Process with translation
    from anytran.pipeline_config import PipelineConfig
    result = processing.process_audio_chunk(
        audio_segment,
        16000,
        PipelineConfig(
            input_lang="en",
            output_lang="fr",
            magnitude_threshold=0.001,
            model="small",
            verbose=False,
            scribe_backend="auto",
            text_translation_target="fr",
            slate_backend="googletrans",
            voice_backend="gtts",
            voice_model=None,
            voice_lang=None,
        ),
        stream_id=None,
        timing_stats=None,
        scribe_tts_segments=scribe_tts_segments,
        slate_tts_segments=slate_tts_segments,
    )
    
    # Should have generated both scribe (English) and slate (French)
    assert len(captured_calls) == 2, \
        f"Expected 2 calls (scribe + slate), got {len(captured_calls)}"
    
    # First call should be scribe (English)
    assert captured_calls[0]['language'] == 'en', \
        f"Expected scribe in English (en), got {captured_calls[0]['language']}"
    assert captured_calls[0]['text'] == "Hello world"
    
    # Second call should be slate (French)
    assert captured_calls[1]['language'] == 'fr', \
        f"Expected slate in French (fr), got {captured_calls[1]['language']}"
    assert captured_calls[1]['text'] == "Bonjour le monde"
    
    # Both segments should have been collected
    assert len(scribe_tts_segments) == 1
    assert len(slate_tts_segments) == 1
