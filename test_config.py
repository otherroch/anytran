#!/usr/bin/env python3

"""Test script to verify the RunnerConfig class works correctly."""

from src.anytran.runners.config import RunnerConfig

# Test creating a config object
config_dict = {
    "input_lang": "en",
    "output_lang": "es",
    "magnitude_threshold": 0.5,
    "model": "gemma4",
    "verbose": True,
    "mqtt_broker": "localhost",
    "mqtt_port": 1883,
    "mqtt_username": "user",
    "mqtt_password": "pass",
    "mqtt_topic": "test/topic",
    "scribe_vad": True,
    "voice_backend": "coqui",
    "voice_model": "tts_model",
    "window_seconds": 5.0,
    "overlap_seconds": 1.0,
    "timers": True,
    "timers_all": False,
    "scribe_backend": "whisper",
    "text_translation_target": "es",
    "slate_backend": "translategemma",
    "voice_lang": "es",
    "scribe_text": "scribe.txt",
    "slate_text": "slate.txt",
    "voice_match": True,
    "keep_temp": False,
    "dedup": True,
    "lang_prefix": False,
    "normalize": True,
    "slate_no_opt": False,
    "scribe_voice": "scribe.wav",
    "slate_voice": "slate.wav",
    "chat_log_dir": "/tmp/chat",
    "capture_voice": "/tmp/capture.wav"
}

# Create a RunnerConfig instance
config = RunnerConfig(**config_dict)

# Test accessing attributes
print("Configuration test:")
print(f"Input language: {config.input_lang}")
print(f"Output language: {config.output_lang}")
print(f"Model: {config.model}")
print(f"MQTT broker: {config.mqtt_broker}")
print(f"Verbose: {config.verbose}")
print(f"Window seconds: {config.window_seconds}")
print(f"Voice backend: {config.voice_backend}")
print(f"Voice model: {config.voice_model}")

print("All configuration attributes accessible successfully!")