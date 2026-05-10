"""Anytran package for real-time audio translation."""

__all__ = ["__version__", "ProcessingConfig", "process_audio_chunk", "build_output_prefix"]
__version__ = "0.1.0"

from .processing import ProcessingConfig, process_audio_chunk, build_output_prefix