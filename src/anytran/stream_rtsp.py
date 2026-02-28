import numpy as np


def stream_rtsp_audio(rtsp_url, audio_queue, sample_rate=16000):
    container = None
    try:
        import av

        options = {
            "rtsp_transport": "tcp",
            "max_delay": "500000",
            "reorder_queue_size": "0",
        }

        container = av.open(rtsp_url, options=options, timeout=10.0)

        audio_stream = None
        for stream in container.streams:
            if stream.type == "audio":
                audio_stream = stream
                break

        if audio_stream is None:
            print(f"No audio stream found in RTSP URL: {rtsp_url}")
            return

        print(f"Connected to RTSP stream: {rtsp_url}")
        print(
            "Audio codec: {0}, Sample rate: {1} Hz, Channels: {2}".format(
                audio_stream.codec_context.name,
                audio_stream.codec_context.sample_rate,
                audio_stream.codec_context.channels,
            )
        )

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)

        for frame in container.decode(audio=0):
            resampled_frames = resampler.resample(frame)
            for resampled_frame in resampled_frames:
                audio_array = resampled_frame.to_ndarray()
                audio_np = audio_array.astype(np.float32) / 32768.0
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()

                if audio_queue.full():
                    try:
                        audio_queue.get_nowait()
                    except Exception:
                        pass
                audio_queue.put(audio_np)

    except ImportError:
        print("PyAV not installed. Please install with: pip install av")
        return
    except Exception as exc:
        print(f"Error in RTSP streaming: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
