class RunnerConfig:
    """Configuration class to group parameters for runner functions."""
    
    def __init__(self, **kwargs):
        # Input
        self.input_lang = kwargs.get('input_lang')
        self.output_lang = kwargs.get('output_lang')
        
        # Stage 1 outputs (English transcription)
        self.scribe_text = kwargs.get('scribe_text')
        self.scribe_voice = kwargs.get('scribe_voice')
        
        # Stage 2 outputs (Translation or re-output of Stage 1)
        self.needs_translation = kwargs.get('needs_translation')
        self.slate_text = kwargs.get('slate_text')
        self.slate_voice = kwargs.get('slate_voice')
        
        # Backend
        self.model = kwargs.get('model')
        self.scribe_backend = kwargs.get('scribe_backend')
        self.magnitude_threshold = kwargs.get('magnitude_threshold')
        
        # Translation
        self.text_translation_target = kwargs.get('text_translation_target')
        self.slate_backend = kwargs.get('slate_backend')
        
        # TTS
        self.voice_lang = kwargs.get('voice_lang')
        self.voice_backend = kwargs.get('voice_backend')
        self.voice_model = kwargs.get('voice_model')
        self.voice_match = kwargs.get('voice_match')
        
        # Audio processing
        self.scribe_vad = kwargs.get('scribe_vad')
        self.window_seconds = kwargs.get('window_seconds')
        self.overlap_seconds = kwargs.get('overlap_seconds')
        
        # MQTT
        self.mqtt_broker = kwargs.get('mqtt_broker')
        self.mqtt_port = kwargs.get('mqtt_port')
        self.mqtt_username = kwargs.get('mqtt_username')
        self.mqtt_password = kwargs.get('mqtt_password')
        self.mqtt_topic = kwargs.get('mqtt_topic')
        
        # Misc
        self.verbose = kwargs.get('verbose')
        self.timers = kwargs.get('timers')
        self.timers_all = kwargs.get('timers_all')
        self.chat_log_dir = kwargs.get('chat_log_dir')
        self.keep_temp = kwargs.get('keep_temp')
        self.dedup = kwargs.get('dedup')
        self.lang_prefix = kwargs.get('lang_prefix')
        self.normalize = kwargs.get('normalize', True)
        self.normalize_input = kwargs.get('normalize_input', True)
        self.slate_no_opt = kwargs.get('slate_no_opt')
        
        # Capture original input voice
        self.capture_voice = kwargs.get('capture_voice')
        
        # File-specific parameters
        self.scribe_text_file = kwargs.get('scribe_text_file')
        self.slate_text_file = kwargs.get('slate_text_file')
        self.output_audio_path = kwargs.get('output_audio_path')
        self.slate_audio_path = kwargs.get('slate_audio_path')
        self.batch = kwargs.get('batch', 0)
        
        # RTSP-specific parameters
        self.rtsp_url = kwargs.get('rtsp_url')
        self.rtsp_urls = kwargs.get('rtsp_urls')
        self.output_device = kwargs.get('output_device')
        self.topic_names = kwargs.get('topic_names')
        
        # YouTube-specific parameters
        self.youtube_url = kwargs.get('youtube_url')
        self.youtube_api_key = kwargs.get('youtube_api_key')
        self.youtube_js_runtime = kwargs.get('youtube_js_runtime')
        self.youtube_remote_components = kwargs.get('youtube_remote_components')
        
        # Additional parameters that were missing
        self.scribe_tts_segments = kwargs.get('scribe_tts_segments')
        self.slate_tts_segments = kwargs.get('slate_tts_segments')
