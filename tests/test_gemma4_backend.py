"""Tests for Gemma4 backend support in scribe, slate, config, and processing."""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from anytran.gemma4_backend import _clean_gemma4_output


# ---------------------------------------------------------------------------
# Output cleanup tests
# ---------------------------------------------------------------------------
class TestCleanGemma4Output(unittest.TestCase):
    """Test _clean_gemma4_output strips artifacts from model responses."""

    def test_returns_none_for_empty(self):
        self.assertIsNone(_clean_gemma4_output(""))
        self.assertIsNone(_clean_gemma4_output(None))

    def test_strips_prompt_echo(self):
        prompt = "Transcribe this audio and translate it to fr."
        text = f"{prompt}\nBonjour le monde."
        self.assertEqual(_clean_gemma4_output(text, prompt_text=prompt), "Bonjour le monde.")

    def test_strips_prompt_echo_at_end(self):
        prompt = "Transcribe this audio and translate it to fr."
        text = f"Bonjour.\n{prompt}"
        self.assertEqual(_clean_gemma4_output(text, prompt_text=prompt), "Bonjour.")

    def test_strips_timestamp_artifacts(self):
        text = "[ 0m0s311ms - 0m1s211ms ] Transcribe this audio and translate it to fr.\nBonjour."
        result = _clean_gemma4_output(text, prompt_text="Transcribe this audio and translate it to fr.")
        self.assertEqual(result, "Bonjour.")

    def test_strips_pure_timestamp_line(self):
        text = "[ 0m0s311ms - 0m1s211ms ]\nBonjour."
        self.assertEqual(_clean_gemma4_output(text), "Bonjour.")

    def test_strips_unable_to_transcribe(self):
        text = "Hello, I'm unable to transcribe that audio.\nBonjour le monde."
        self.assertEqual(_clean_gemma4_output(text), "Bonjour le monde.")

    def test_strips_sorry_unable(self):
        text = "Sorry, I'm unable to transcribe that audio."
        self.assertIsNone(_clean_gemma4_output(text))

    def test_strips_music_markers(self):
        text = "[Music]\nBonjour.\n[music] [music]"
        self.assertEqual(_clean_gemma4_output(text), "Bonjour.")

    def test_strips_emoji_music_markers(self):
        text = "[ 🎵 ]\nBonjour."
        self.assertEqual(_clean_gemma4_output(text), "Bonjour.")

    def test_strips_inline_music_markers(self):
        text = "[Music] **French:** [Music]"
        # After stripping music markers and label, nothing or minimal remains
        result = _clean_gemma4_output(text)
        # Should be None since nothing useful remains after stripping
        self.assertIsNone(result)

    def test_strips_formatting_labels(self):
        text = "**French Translation:** J'étais situé dans le quartier."
        self.assertEqual(
            _clean_gemma4_output(text),
            "J'étais situé dans le quartier.",
        )

    def test_strips_plain_translation_label(self):
        text = "French translation: Bonjour le monde."
        self.assertEqual(_clean_gemma4_output(text), "Bonjour le monde.")

    def test_preserves_clean_text(self):
        text = "Bonjour le monde."
        self.assertEqual(_clean_gemma4_output(text), "Bonjour le monde.")

    def test_all_artifacts_returns_none(self):
        text = "[Music]\n[Music] [Music]\n[ 🎵 ]"
        self.assertIsNone(_clean_gemma4_output(text))

    def test_mixed_real_output_example(self):
        """Simulate the kind of mixed output from the 4B model."""
        prompt = "Transcribe this audio and translate it to fr."
        text = (
            "Transcribe this audio and translate it to fr.\n"
            "**French Translation:** Nous avons terminé notre journée avec un.\n"
            "[Music] [Music]"
        )
        result = _clean_gemma4_output(text, prompt_text=prompt)
        self.assertEqual(result, "Nous avons terminé notre journée avec un.")

    def test_2b_model_artifacts(self):
        """Simulate 2B model output with timestamps and apologies."""
        prompt = "Transcribe this audio and translate it to fr."
        text = (
            "[ 0m0s311ms - 0m1s211ms ] Transcribe this audio and translate it to fr.\n"
            "Sorry, I'm unable to transcribe that audio.\n"
            "Nous sommes arrivés à Windsor tôt."
        )
        result = _clean_gemma4_output(text, prompt_text=prompt)
        self.assertEqual(result, "Nous sommes arrivés à Windsor tôt.")

    # -------------------------------------------------------------------
    # Translated prompt echo detection
    # -------------------------------------------------------------------

    def test_strips_french_prompt_echo(self):
        """Catches French translation of the English instruction prompt."""
        text = (
            "Écoutez ceci et traduisez-le en français.\n"
            "Nous avons décidé de visiter Londres."
        )
        self.assertEqual(
            _clean_gemma4_output(text),
            "Nous avons décidé de visiter Londres.",
        )

    def test_strips_repeated_french_prompt_echoes(self):
        """Multiple translated echoes scattered through the output."""
        text = (
            "Écoutez ceci et traduisez-le en français.\n"
            "Pourquoi ne pas aller à Londres en février?\n"
            "Écoutez ceci et traduisez-le en français.\n"
            "Le lendemain, nous avons passé la journée à visiter."
        )
        result = _clean_gemma4_output(text)
        self.assertEqual(
            result,
            "Pourquoi ne pas aller à Londres en février?\n"
            "Le lendemain, nous avons passé la journée à visiter.",
        )

    def test_strips_spanish_prompt_echo(self):
        """Catches Spanish translation of the instruction prompt."""
        text = "Escucha esto y tradúcelo al español.\nHola mundo."
        self.assertEqual(_clean_gemma4_output(text), "Hola mundo.")

    def test_strips_german_prompt_echo(self):
        """Catches German translation of the instruction prompt."""
        text = "Hören Sie sich das an und übersetzen Sie es auf Deutsch.\nHallo Welt."
        self.assertEqual(_clean_gemma4_output(text), "Hallo Welt.")

    def test_strips_italian_prompt_echo(self):
        """Catches Italian translation of the instruction prompt."""
        text = "Ascolta questo e traducilo in italiano.\nCiao mondo."
        self.assertEqual(_clean_gemma4_output(text), "Ciao mondo.")

    def test_strips_translate_audio_pattern(self):
        """Catches 'translate the audio' pattern."""
        text = "Traduisez l'audio en français.\nBonjour le monde."
        self.assertEqual(_clean_gemma4_output(text), "Bonjour le monde.")

    def test_strips_transcribe_audio_pattern(self):
        """Catches 'transcribe this audio' pattern."""
        text = "Transcribe this audio.\nHello world."
        self.assertEqual(_clean_gemma4_output(text), "Hello world.")

    def test_strips_meta_instruction_output_only(self):
        """Catches 'output only the translation' meta-instruction leak."""
        text = "Output only the translated text.\nBonjour."
        self.assertEqual(_clean_gemma4_output(text), "Bonjour.")

    def test_strips_meta_instruction_reply_with(self):
        """Catches 'reply with only the translation' meta-instruction leak."""
        text = "Reply with only the translation.\nBonjour."
        self.assertEqual(_clean_gemma4_output(text), "Bonjour.")

    def test_strips_do_not_repeat_leak(self):
        """Catches 'do not repeat these instructions' meta-instruction leak."""
        text = "Do not repeat these instructions.\nBonjour le monde."
        self.assertEqual(_clean_gemma4_output(text), "Bonjour le monde.")

    # -------------------------------------------------------------------
    # Translated apology detection
    # -------------------------------------------------------------------

    def test_strips_french_apology_desole(self):
        """Catches French 'désolé... écouter l'audio' apology."""
        text = (
            "Bonjour, je suis désolé, je n'ai pas pu écouter l'audio.\n"
            "Nous avons visité le château."
        )
        self.assertEqual(
            _clean_gemma4_output(text),
            "Nous avons visité le château.",
        )

    def test_strips_french_apology_pas_pu(self):
        """Catches French 'pas pu transcrire' apology."""
        text = "Je n'ai pas pu transcrire cet audio.\nBonjour."
        self.assertEqual(_clean_gemma4_output(text), "Bonjour.")

    def test_strips_spanish_apology(self):
        """Catches Spanish apology about audio."""
        text = "Lo siento, no pude escuchar el audio.\nHola mundo."
        self.assertEqual(_clean_gemma4_output(text), "Hola mundo.")

    def test_strips_german_apology(self):
        """Catches German apology about audio."""
        text = "Es tut mir leid, ich kann nicht das Audio transkribieren.\nHallo Welt."
        self.assertEqual(_clean_gemma4_output(text), "Hallo Welt.")

    # -------------------------------------------------------------------
    # Real-world combined scenario from user report
    # -------------------------------------------------------------------

    def test_real_world_2b_after_output(self):
        """Test with output that matches the user-reported 2B model artifacts."""
        text = (
            "Écoutez ceci et traduisez-le en français.\n"
            "Comme nous n'avons pas eu assez de pluie, nous avons décidé\n"
            "Pourquoi ne pas aller à Londres en février?\n"
            "Écoutez ceci et traduisez-le en français.\n"
            "Je suis situé dans le Seven Dials Quarter.\n"
            "Bonjour, je suis désolé, je n'ai pas pu écouter l'audio.\n"
            "Écoutez ceci et traduisez-le en français.\n"
        )
        result = _clean_gemma4_output(text)
        self.assertEqual(
            result,
            "Comme nous n'avons pas eu assez de pluie, nous avons décidé\n"
            "Pourquoi ne pas aller à Londres en février?\n"
            "Je suis situé dans le Seven Dials Quarter.",
        )

    def test_preserves_content_with_listen_word(self):
        """Should NOT strip lines that merely mention listening as content."""
        text = "J'ai écouté la musique hier soir."
        # This contains "écouté" but no translate verb, so it should be kept
        self.assertEqual(_clean_gemma4_output(text), "J'ai écouté la musique hier soir.")

    def test_preserves_content_with_translate_word(self):
        """Should NOT strip lines that mention translation as content."""
        text = "Le traducteur a traduit le livre en anglais."
        # Contains "traduit" but no listen/audio verb combo that matches echo
        self.assertEqual(
            _clean_gemma4_output(text),
            "Le traducteur a traduit le livre en anglais.",
        )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
class TestGemma4Config(unittest.TestCase):
    """Test Gemma4 configuration in config.py."""

    def setUp(self):
        import anytran.config as cfg
        self._orig = dict(cfg._gemma4_config)

    def tearDown(self):
        import anytran.config as cfg
        cfg._gemma4_config.update(self._orig)

    def test_default_model_name(self):
        from anytran.config import get_gemma4_config
        self.assertEqual(get_gemma4_config()["model_name"], "google/gemma-4-E4B-it")

    def test_set_model_name(self):
        from anytran.config import set_gemma4_config, get_gemma4_config
        set_gemma4_config(model_name="google/gemma-4-E2B-it")
        self.assertEqual(get_gemma4_config()["model_name"], "google/gemma-4-E2B-it")

    def test_set_none_keeps_existing(self):
        from anytran.config import set_gemma4_config, get_gemma4_config
        set_gemma4_config(model_name="custom/model")
        set_gemma4_config(model_name=None)
        self.assertEqual(get_gemma4_config()["model_name"], "custom/model")


# ---------------------------------------------------------------------------
# Text translator config + routing tests
# ---------------------------------------------------------------------------
class TestGemma4TextConfig(unittest.TestCase):
    """Test set_gemma4_text_config and routing in text_translator."""

    def setUp(self):
        import anytran.text_translator as tt
        self._orig_backend = tt._translation_backend
        self._orig_model = tt._gemma4_text_model_name

    def tearDown(self):
        import anytran.text_translator as tt
        tt._translation_backend = self._orig_backend
        tt._gemma4_text_model_name = self._orig_model

    def test_set_gemma4_text_config(self):
        import anytran.text_translator as tt
        from anytran.text_translator import set_gemma4_text_config
        set_gemma4_text_config("google/gemma-4-E2B-it")
        self.assertEqual(tt._gemma4_text_model_name, "google/gemma-4-E2B-it")

    def test_translate_text_routes_to_gemma4(self):
        """Verify translate_text dispatches to translate_text_gemma4."""
        with patch("anytran.text_translator.translate_text_gemma4", return_value="Bonjour") as mock:
            from anytran.text_translator import translate_text
            result = translate_text("Hello", "en", "fr", backend="gemma4")
            mock.assert_called_once_with("Hello", "en", "fr", False)
            self.assertEqual(result, "Bonjour")

    def test_translate_text_gemma4_returns_none_without_deps(self):
        """translate_text_gemma4 returns None when transformers/torch unavailable."""
        import anytran.text_translator as tt
        orig_transformers = tt._TRANSFORMERS_AVAILABLE
        orig_torch = tt._TORCH_AVAILABLE
        try:
            tt._TRANSFORMERS_AVAILABLE = False
            tt._TORCH_AVAILABLE = False
            from anytran.text_translator import translate_text_gemma4
            result = translate_text_gemma4("Hello", "en", "fr", verbose=True)
            self.assertIsNone(result)
        finally:
            tt._TRANSFORMERS_AVAILABLE = orig_transformers
            tt._TORCH_AVAILABLE = orig_torch


# ---------------------------------------------------------------------------
# Whisper backend dispatch tests
# ---------------------------------------------------------------------------
class TestGemma4ScribeBackendDispatch(unittest.TestCase):
    """Test that get_effective_backend and translate_audio handle gemma4."""

    def test_get_effective_backend_gemma4(self):
        from anytran.whisper_backend import get_effective_backend
        self.assertEqual(get_effective_backend("gemma4"), "gemma4")

    def test_translate_audio_dispatches_to_gemma4(self):
        """translate_audio should call translate_audio_gemma4 when backend is gemma4."""
        audio = np.zeros(16000, dtype=np.float32)
        with patch("anytran.gemma4_backend.translate_audio_gemma4", return_value=(audio, "hello", "en")) as mock:
            from anytran.whisper_backend import translate_audio
            result = translate_audio(
                audio, samplerate=16000, backend_preference="gemma4"
            )
            mock.assert_called_once()
            self.assertEqual(result[1], "hello")


# ---------------------------------------------------------------------------
# CLI choices tests
# ---------------------------------------------------------------------------
class TestGemma4CLIChoices(unittest.TestCase):
    """Verify gemma4 is an accepted CLI choice."""

    def _get_parser_choices(self, dest):
        """Return the choices set for a given argument dest."""
        import argparse

        parser = argparse.ArgumentParser()
        scribe_group = parser.add_argument_group("scribe")
        scribe_group.add_argument("--scribe-backend", type=str, choices=[
            "whispercpp", "whispercpp-cli", "faster-whisper", "whisper-ctranslate2", "gemma4"
        ])
        slate_group = parser.add_argument_group("slate")
        slate_group.add_argument("--slate-backend", type=str, choices=[
            "googletrans", "libretranslate", "translategemma", "metanllb", "marianmt", "gemma4", "none"
        ])
        for action in parser._actions:
            if action.dest == dest:
                return action.choices
        return None

    def test_scribe_backend_accepts_gemma4(self):
        choices = self._get_parser_choices("scribe_backend")
        self.assertIn("gemma4", choices)

    def test_slate_backend_accepts_gemma4(self):
        choices = self._get_parser_choices("slate_backend")
        self.assertIn("gemma4", choices)


# ---------------------------------------------------------------------------
# One-pass optimization in processing
# ---------------------------------------------------------------------------
class TestGemma4OnePass(unittest.TestCase):
    """Test the gemma4 one-pass optimization in process_audio_chunk."""

    def test_one_pass_skips_stage2(self):
        """When both backends are gemma4 with same model, stage 2 is skipped."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        combined_result = (audio, "Bonjour le monde", None)

        with patch("anytran.processing.translate_audio_gemma4_combined", return_value=combined_result) as mock_combined, \
             patch("anytran.processing.translate_audio") as mock_translate, \
             patch("anytran.processing.translate_text") as mock_text, \
             patch("anytran.processing.get_gemma4_config", return_value={"model_name": "google/gemma-4-E4B-it"}), \
             patch("anytran.text_translator._gemma4_text_model_name", "google/gemma-4-E4B-it"):
            from anytran.processing import process_audio_chunk
            from anytran.pipeline_config import PipelineConfig
            cfg = PipelineConfig(
                output_lang="fr",
                magnitude_threshold=0.001,
                scribe_backend="gemma4",
                slate_backend="gemma4",
                text_translation_target="fr",
            )
            result = process_audio_chunk(
                audio_segment=audio,
                rate=16000,
                config=cfg,
            )
            # One-pass function should be called
            mock_combined.assert_called_once()
            # Regular stage1 and stage2 should NOT be called
            mock_translate.assert_not_called()
            mock_text.assert_not_called()
            # Result should contain the combined text
            self.assertIsNotNone(result)
            self.assertIn("Bonjour le monde", result["slate"])

    def test_different_models_no_one_pass(self):
        """When scribe and slate use different gemma4 models, one-pass should NOT activate."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        stage1_result = (audio, "Hello world", "en")
        stage2_result = "Bonjour le monde"

        with patch("anytran.processing.translate_audio_gemma4_combined") as mock_combined, \
             patch("anytran.processing.translate_audio", return_value=stage1_result), \
             patch("anytran.processing.translate_text", return_value=stage2_result), \
             patch("anytran.processing.get_gemma4_config", return_value={"model_name": "google/gemma-4-E4B-it"}), \
             patch("anytran.text_translator._gemma4_text_model_name", "google/gemma-4-E2B-it"):
            from anytran.processing import process_audio_chunk
            from anytran.pipeline_config import PipelineConfig
            cfg = PipelineConfig(
                output_lang="fr",
                magnitude_threshold=0.001,
                scribe_backend="gemma4",
                slate_backend="gemma4",
                text_translation_target="fr",
            )
            result = process_audio_chunk(
                audio_segment=audio,
                rate=16000,
                config=cfg,
            )
            # One-pass should NOT be called because models differ
            mock_combined.assert_not_called()

    def test_no_one_pass_when_target_is_english(self):
        """When target is English, one-pass should NOT activate."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        stage1_result = (audio, "Hello", "en")

        with patch("anytran.processing.translate_audio_gemma4_combined") as mock_combined, \
             patch("anytran.processing.translate_audio", return_value=stage1_result), \
             patch("anytran.processing.get_gemma4_config", return_value={"model_name": "google/gemma-4-E4B-it"}), \
             patch("anytran.text_translator._gemma4_text_model_name", "google/gemma-4-E4B-it"):
            from anytran.processing import process_audio_chunk
            from anytran.pipeline_config import PipelineConfig
            cfg = PipelineConfig(
                output_lang="en",
                magnitude_threshold=0.001,
                scribe_backend="gemma4",
                slate_backend="gemma4",
                text_translation_target="en",
            )
            result = process_audio_chunk(
                audio_segment=audio,
                rate=16000,
                config=cfg,
            )
            # One-pass should NOT be called because target is English
            mock_combined.assert_not_called()


# ---------------------------------------------------------------------------
# _configure_backends tests
# ---------------------------------------------------------------------------
class TestConfigureBackendsGemma4(unittest.TestCase):
    """Test _configure_backends handles gemma4 for scribe and slate."""

    def test_configure_gemma4_scribe(self):
        """Verify scribe gemma4 configuration."""
        import argparse
        args = argparse.Namespace(
            input=None,
            scribe_backend="gemma4",
            scribe_model="google/gemma-4-E2B-it",
            slate_backend="googletrans",
            slate_model=None,
            libretranslate_url=None,
            whispercpp_cli_detect_lang=False,
            whispercpp_bin=None,
            whispercpp_model_dir="./models",
            whispercpp_threads=4,
            no_auto_download=False,
            whisper_ctranslate2_device="auto",
            whisper_ctranslate2_device_index=None,
            whisper_ctranslate2_compute_type="default",
        )
        with patch("anytran.main.set_whisper_backend") as mock_wb, \
             patch("anytran.main.set_gemma4_config") as mock_gc, \
             patch("anytran.main.set_translation_backend"), \
             patch("anytran.main.set_whispercpp_cli_detect_lang"):
            from anytran.main import _configure_backends
            _configure_backends(args)
            mock_wb.assert_called_with("gemma4")
            mock_gc.assert_called_with("google/gemma-4-E2B-it")

    def test_configure_gemma4_slate(self):
        """Verify slate gemma4 configuration."""
        import argparse
        args = argparse.Namespace(
            input="test.txt",
            scribe_backend="faster-whisper",
            scribe_model="medium",
            slate_backend="gemma4",
            slate_model="google/gemma-4-E4B-it",
            libretranslate_url=None,
            whispercpp_cli_detect_lang=False,
        )
        with patch("anytran.main.set_translation_backend") as mock_tb, \
             patch("anytran.main.set_gemma4_text_config") as mock_gtc:
            from anytran.main import _configure_backends
            _configure_backends(args)
            mock_tb.assert_called_with("gemma4")
            mock_gtc.assert_called_with("google/gemma-4-E4B-it")

    def test_configure_gemma4_slate_default_model(self):
        """When slate_model is None, default model should be used."""
        import argparse
        args = argparse.Namespace(
            input="test.txt",
            scribe_backend="faster-whisper",
            scribe_model="medium",
            slate_backend="gemma4",
            slate_model=None,
            libretranslate_url=None,
            whispercpp_cli_detect_lang=False,
        )
        with patch("anytran.main.set_translation_backend"), \
             patch("anytran.main.set_gemma4_text_config") as mock_gtc:
            from anytran.main import _configure_backends
            _configure_backends(args)
            mock_gtc.assert_called_with("google/gemma-4-E4B-it")


if __name__ == "__main__":
    unittest.main()
