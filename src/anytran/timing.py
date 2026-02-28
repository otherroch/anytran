import threading
import time


def add_timing(timings, label, start_time):
    if timings is None:
        return
    timings.append((label, time.perf_counter() - start_time))


def format_timing(timings):
    if not timings:
        return ""
    return ", ".join(f"{label}={elapsed * 1000:.1f}ms" for label, elapsed in timings)


class TimingsAggregator:
    def __init__(self, label=""):
        self.label = label
        self._lock = threading.Lock()
        self._stats = {}

    def add_time(self, label, start_time, end_time):
        """Add a single timing entry from explicit start and end times."""
        elapsed = end_time - start_time
        with self._lock:
            if label not in self._stats:
                self._stats[label] = {"count": 0, "total": 0.0, "min": None, "max": None}
            entry = self._stats[label]
            entry["count"] += 1
            entry["total"] += elapsed
            entry["min"] = elapsed if entry["min"] is None else min(entry["min"], elapsed)
            entry["max"] = elapsed if entry["max"] is None else max(entry["max"], elapsed)

    def add(self, timings, prefix=None):
        if not timings:
            return
        with self._lock:
            for label, elapsed in timings:
                key = f"{prefix}.{label}" if prefix else label
                if key not in self._stats:
                    self._stats[key] = {"count": 0, "total": 0.0, "min": None, "max": None}
                entry = self._stats[key]
                entry["count"] += 1
                entry["total"] += elapsed
                entry["min"] = elapsed if entry["min"] is None else min(entry["min"], elapsed)
                entry["max"] = elapsed if entry["max"] is None else max(entry["max"], elapsed)

    def format_summary(self):
        with self._lock:
            if not self._stats:
                return ""
            parts = []
            for key in sorted(self._stats.keys()):
                entry = self._stats[key]
                avg = entry["total"] / entry["count"] if entry["count"] else 0.0
                parts.append(
                    f"{key}: total={entry['total']:.3f}s, avg={avg * 1000:.1f}ms, min={entry['min'] * 1000:.1f}ms, max={entry['max'] * 1000:.1f}ms"
                )
            return " | ".join(parts)

    def format_translate_overhead(self, backend_label):
        with self._lock:
            if not self._stats:
                return ""
            total_key = "chunk.translate_total"
            transcribe_key = f"{backend_label}.transcribe"
            if total_key not in self._stats or transcribe_key not in self._stats:
                return ""
            total = self._stats[total_key]["total"]
            transcribe = self._stats[transcribe_key]["total"]
            overhead = max(0.0, total - transcribe)
            return f"translate_overhead: total={overhead:.3f}s (translate_total - {transcribe_key})"

    def format_stage_summary(self):
        """
        Format timing summary with total time, stage times, and percentages.
        
        Returns a formatted string showing:
        - Total processing time
        - Stage 1 (Transcription) time and percentage
        - Stage 2 (Translation) time and percentage
        - Stage 3 (TTS) time and percentage
        """
        with self._lock:
            if not self._stats:
                return ""
            
            # Collect stage times
            stage1_time = 0.0
            stage2_time = 0.0
            stage3_time = 0.0
            
            # Stage 1: Transcription (Whisper)
            if "chunk.stage1_transcription" in self._stats:
                stage1_time = self._stats["chunk.stage1_transcription"]["total"]
            
            # Stage 2: Translation (text-to-text)
            if "chunk.stage2_translation" in self._stats:
                stage2_time = self._stats["chunk.stage2_translation"]["total"]
            
            # Stage 3: TTS (synthesis + playback)
            if "chunk.stage3_tts_synthesis" in self._stats:
                stage3_time += self._stats["chunk.stage3_tts_synthesis"]["total"]
            if "chunk.stage3_tts_playback" in self._stats:
                stage3_time += self._stats["chunk.stage3_tts_playback"]["total"]
            
            # Calculate total
            total_time = stage1_time + stage2_time + stage3_time
            
            if total_time == 0:
                return ""
            
            # Calculate percentages
            stage1_pct = stage1_time / total_time * 100
            stage2_pct = stage2_time / total_time * 100
            stage3_pct = stage3_time / total_time * 100
            
            # Format output
            lines = [
                f"Total: {total_time:.3f}s",
                f"  Stage 1 (Transcription): {stage1_time:.3f}s ({stage1_pct:.1f}%)",
                f"  Stage 2 (Translation):   {stage2_time:.3f}s ({stage2_pct:.1f}%)",
                f"  Stage 3 (TTS):           {stage3_time:.3f}s ({stage3_pct:.1f}%)",
            ]
            
            return "\n".join(lines)
