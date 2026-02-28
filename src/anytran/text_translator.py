"""
Text translation module supporting multiple backends for translating 
English text to non-English languages.

Supports:
- Google Translate (via googletrans)
- LibreTranslate (local/self-hosted)
- TranslateGemma (local AI model)
- MarianMT (Helsinki-NLP Marian machine translation models)
- MetaNLLB (Meta's NLLB local AI model)
- Direct passthrough (no translation)
"""

import os
import time
from typing import Optional, Tuple

try:
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        pipeline,
        AutoTokenizer,
        M2M100ForConditionalGeneration,
        AutoModelForSeq2SeqLM,
        NllbTokenizer,
        NllbTokenizerFast,
        MarianMTModel,
        MarianTokenizer,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoProcessor = AutoModelForImageTextToText = pipeline = None
    AutoTokenizer = M2M100ForConditionalGeneration = AutoModelForSeq2SeqLM = None
    NllbTokenizer = NllbTokenizerFast = MarianMTModel = MarianTokenizer = None

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from googletrans import Translator
    _GOOGLETRANS_AVAILABLE = True
except ImportError:
    Translator = None  # type: ignore[assignment,misc]
    _GOOGLETRANS_AVAILABLE = False

# Global configuration
_translation_backend = "googletrans"
_libretranslate_url = None
_googletrans_translator = None  # Cached googletrans Translator instance to avoid repeated initialization

_translategemma_model = None  # Cached TranslateGemma model
_translategemma_tokenizer = None  # Cached TranslateGemma tokenizer
_translategemma_model_name = "google/translategemma-4b-it"  # Default model

# --- Pipeline version toggle ---
USE_PIPELINE_TRANSLATEGEMMA = False  # Set to True to use the pipeline-based version

_metanllb_model = None  # Cached MetaNLLB model
_metanllb_tokenizer = None  # Cached MetaNLLB tokenizer
_metanllb_model_name = "facebook/nllb-200-1.3B"  # Default MetaNLLB model

_marianmt_model = None  # Cached MarianMT model
_marianmt_tokenizer = None  # Cached MarianMT tokenizer
_marianmt_model_name = None  # If None, auto-derive from language pair as Helsinki-NLP/opus-mt-{src}-{tgt}
_marianmt_loaded_model_name = None  # Track which model name is currently loaded

# Mapping from common ISO 639-1 language codes to FLORES-200 codes used by NLLB
_NLLB_LANG_MAP = {
    "af": "afr_Latn",
    "ar": "arb_Arab",
    "az": "azj_Latn",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "ga": "gle_Latn",
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "mt": "mlt_Latn",
    "my": "mya_Mymr",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "no": "nob_Latn",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "th": "tha_Thai",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "xh": "xho_Latn",
    "zh": "zho_Hans",
    "zu": "zul_Latn",
}


def set_translation_backend(backend: str):
    """Set the text translation backend."""
    global _translation_backend
    _translation_backend = backend


def get_translation_backend() -> str:
    """Get the current text translation backend."""
    return _translation_backend


def set_libretranslate_config(url: str):
    """Set LibreTranslate API URL."""
    global _libretranslate_url
    _libretranslate_url = url


def set_translategemma_config(model_name: str):
    """Set TranslateGemma model name."""
    global _translategemma_model_name
    _translategemma_model_name = model_name


def set_metanllb_config(model_name: str):
    """Set MetaNLLB model name."""
    global _metanllb_model_name
    _metanllb_model_name = model_name


def set_marianmt_config(model_name: str):
    """Set MarianMT model name."""
    global _marianmt_model_name
    _marianmt_model_name = model_name


def _get_googletrans_translator():
    """Create and reuse a single googletrans Translator instance."""
    global _googletrans_translator
    if not _GOOGLETRANS_AVAILABLE:
        raise ImportError(
            "googletrans not installed. Install with: pip install googletrans==4.0.0-rc1"
        )
    if _googletrans_translator is None:
        _googletrans_translator = Translator()
    return _googletrans_translator


def translate_text_googletrans(text: str, source_lang: str, target_lang: str, verbose: bool = False) -> Optional[str]:
    """
    Translate text using googletrans library with retry logic.

    Args:
        text: Text to translate
        source_lang: Source language code (e.g., 'en')
        target_lang: Target language code (e.g., 'es', 'fr', 'de')
        verbose: Print debug information

    Returns:
        Translated text or None on failure
    """
    if not _GOOGLETRANS_AVAILABLE:
        if verbose:
            print("googletrans not installed. Install with: pip install googletrans==4.0.0-rc1")
        return None

    max_retries = 5

    def _is_transient_error(exc: Exception) -> bool:
        """Heuristic to detect transient/network/rate-limit errors based on the message."""
        msg = str(exc).lower()
        transient_markers = [
            "429",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "temporarily overloaded",
            "timeout",
            "timed out",
            "connection aborted",
            "connection reset",
            "connection refused",
            "connection error",
            "network is unreachable",
            "server error",
        ]
        return any(marker in msg for marker in transient_markers)

    for attempt in range(max_retries):
        try:
            translator = _get_googletrans_translator()
            if verbose:
                short_text = "empty text"
                if len(text) > 0:
                    short_text = (text[:10] + '...') if len(text) > 10 else text
                print(f"Googletrans: Translating '{short_text}' from {source_lang} to {target_lang} (attempt {attempt + 1}/{max_retries})...")  
            result = translator.translate(text, src=source_lang, dest=target_lang)
            if result is None or not hasattr(result, 'text') or result.text is None:
                if verbose:
                    short_text = "empty text"
                    if len(text) > 0:
                        short_text = (text[:10] + '...') if len(text) > 10 else text
                    print(f"Googletrans returned None or invalid result for input: '{short_text}' ")
                return None  # Treat None or missing 'text' as a failure   
            if verbose:
                print(f"Googletrans raw result: {result} (type: {type(result)})")       
            translated = result.text
            if verbose and translated is not None:
                print(f"Googletrans translated text: '{translated}'")
            if verbose:
               if len(text) > 10:
                     short_text = text[:10] + "..."
               else:
                     short_text = text
               if len(translated) > 10:
                     short_translated = translated[:10] + "..."
               else:
                     short_translated = translated
               print(f"Googletrans: '{short_text}' -> '{short_translated}' ({source_lang}->{target_lang})")
            return translated
        except AttributeError as exc:
            # Handle httpcore compatibility issue
            error_str = str(exc)
            if "SyncHTTPTransport" in error_str or "httpcore" in error_str.lower():
                if verbose:
                    print("Failed after all retry attempts")
                return None
            # Other AttributeErrors are treated as non-transient errors; do not re-raise
            if verbose:
                print(f"Googletrans non-httpcore AttributeError encountered: {exc}")
            return None
        except Exception as exc:
            # Catch-all for other exceptions; retry on likely transient errors (e.g., HTTP 429, network issues)
            is_last_attempt = attempt == max_retries - 1
            if not is_last_attempt and _is_transient_error(exc):
                backoff_seconds = 2 ** attempt
                if verbose:
                    print(
                        f"Googletrans transient error (attempt {attempt + 1}/{max_retries}): {exc}. "
                         f"Retrying in {backoff_seconds} seconds..."
                    )
                time.sleep(backoff_seconds)
                continue
            if verbose:
                if _is_transient_error(exc):
                    print(
                         f"Googletrans translation failed after {max_retries} attempts due to transient errors: {exc}"
                      )
                else:
                   print(f"Googletrans translation failed: {exc}")
                   short_text = "empty text"
                   if len(text) > 0:
                    short_text = (text[:10] + '...') if len(text) > 10 else text
                    print(f"Googletrans translation failed with non-transient error for input: '{short_text}'")
            return None


def translate_text_libretranslate(text: str, source_lang: str, target_lang: str, verbose: bool = False) -> Optional[str]:
    """
    Translate text using LibreTranslate API.

    Args:
        text: Text to translate
        source_lang: Source language code (e.g., 'en')
        target_lang: Target language code (e.g., 'es', 'fr', 'de')
        verbose: Print debug information

    Returns:
        Translated text or None on failure
    """
    if not _libretranslate_url:
        if verbose:
            print("LibreTranslate URL not configured. Use --libretranslate-url")
        return None

    try:
        import requests
        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        }
        response = requests.post(f"{_libretranslate_url}/translate", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        translated = result.get("translatedText", "")
        if verbose:
            print(f"LibreTranslate: '{text}' -> '{translated}' ({source_lang}->{target_lang})")
        return translated
    except ImportError:
        if verbose:
            print("requests not installed. Install with: pip install requests")
        return None
    except Exception as exc:
        if verbose:
            print(f"LibreTranslate translation failed: {exc}")
        return None


def get_translategemma_model():
    processor, model = _get_translategemma_model()
    return processor, model


def _get_translategemma_model():
    """Load and cache TranslateGemma model and tokenizer."""
    global _translategemma_model, _translategemma_tokenizer
    if not _TRANSFORMERS_AVAILABLE or not _TORCH_AVAILABLE:
        raise ImportError(
            "TranslateGemma requires transformers and torch. "
            "Install with: pip install transformers torch"
        )
    if _translategemma_model is None or _translategemma_tokenizer is None:
        try:

            model_name = _translategemma_model_name
            if model_name.startswith("google/"):
                print(f"TranslateGemma: Using Google Gemma model '{model_name}'")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"TranslateGemma: Loading processor '{model_name}' on device '{device}'")
            _translategemma_tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
            print(f"TranslateGemma: Loading model '{model_name}' on device '{device}'")
            _translategemma_model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            _translategemma_model = _translategemma_model.to(device)
        except Exception as e:
            print(f"TranslateGemma: Failed to load model or processor: {e}")
            raise
    
        print(f"TranslateGemma: Loaded model '{_translategemma_model_name}' on device '{_translategemma_model.device}'")
    return _translategemma_model, _translategemma_tokenizer



def translate_text_translategemma_pipeline(text: str, source_lang: str, target_lang: str, verbose: bool = False) -> Optional[str]:
    """
    Translate text using the pipeline-based TranslateGemma implementation (from pipeline.py example).
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        pipe = pipeline(
            "image-text-to-text",
            model=_translategemma_model_name,
            device=device,
            dtype=dtype
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text,
                    }
                ],
            }
        ]
        output = pipe(text=messages, max_new_tokens=200)
        # The output format is: output[0]["generated_text"][-1]["content"]
        translated = output[0]["generated_text"][-1]["content"]
        if verbose:
            print(f"Pipeline TranslateGemma: '{text[:30]}' -> '{translated[:30]}' ({source_lang}->{target_lang})")
        return translated
    except Exception as exc:
        if verbose:
            print(f"Pipeline TranslateGemma translation failed: {exc}")
        return None


def translate_text_translategemma(text: str, source_lang: str, target_lang: str, verbose: bool = False) -> Optional[str]:
    """
    Translate text using TranslateGemma (local Gemma-based translation model).
    Uses pipeline-based version if USE_PIPELINE_TRANSLATEGEMMA is True.
    """
    if USE_PIPELINE_TRANSLATEGEMMA:
        return translate_text_translategemma_pipeline(text, source_lang, target_lang, verbose)
    try:
        model, processor = _get_translategemma_model()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text,
                    }
                ],
            }
        ]
        if verbose:
            print(f"TranslateGemma: Loaded model and processor: '{_translategemma_model_name}'")
            print(f"TranslateGemma: Translating '{text[:30]}' from {source_lang} to {target_lang}...")
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        input_len = len(inputs['input_ids'][0])
        if verbose:
            print(f"TranslateGemma: Input tokens len: {input_len}")
        with torch.inference_mode():
            generation = model.generate(**inputs, do_sample=False, max_new_tokens=1000)
        generation = generation[0][input_len:]
        if verbose:
            print(f"TranslateGemma: Generation tokens len: {len(generation)}")
        translated = processor.decode(generation, skip_special_tokens=True)
        if verbose and translated:
            short_text = (text[:30] + '...') if len(text) > 30 else text
            short_translated = (translated[:30] + '...') if len(translated) > 30 else translated
            print(f"TranslateGemma: '{short_text}' -> '{short_translated}' ({source_lang}->{target_lang})")
        return translated
    except ImportError as e:
        if verbose:
            print(f"TranslateGemma dependencies not installed: {e}")


def get_metanllb_model():
    """
    Ensure the MetaNLLB model and tokenizer are loaded and return them.
    """
    
    
    global _metanllb_model, _metanllb_tokenizer, _metanllb_model_name

    # If the model and tokenizer are already loaded, reuse the cached instances.
    if _metanllb_model is not None and _metanllb_tokenizer is not None:
        return
        
    _get_metanllb_model()
 

def _get_metanllb_model(source_lang=None, target_lang=None, verbose=False):
    global _metanllb_model, _metanllb_tokenizer, _metanllb_model_name

    if not _TRANSFORMERS_AVAILABLE or not _TORCH_AVAILABLE:
        raise ImportError(
            "MetaNLLB requires transformers and torch. "
            "Install with: pip install transformers torch"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"MetaNLLB: Loading tokenizer '{_metanllb_model_name}', src_lang='{source_lang}', tgt_lang='{target_lang}' on device '{device}' ...")
    
    #  use_fast=False
# legacy_behaviour=True,
    _metanllb_tokenizer = NllbTokenizerFast.from_pretrained(_metanllb_model_name,
                                                        src_lang=source_lang, 
                                                        tgt_lang=target_lang,
                                                        use_fast=True,
    #                                                    legacy_behaviour=True,
                                                        )
        

    #_metanllb_tokenizer = AutoTokenizer.from_pretrained(_metanllb_model_name,use_fast=True, trust_remote_code=True)
    
    #                                                    src_lang=source_lang,
    #                                                    use_fast=False
    #                                                    )
    
    print(f"MetaNLLB: Loading model '{_metanllb_model_name}' on device '{device}'")
    # config = AutoConfig.from_pretrained(_metanllb_model_name, trust_remote_code=True)
    
    _metanllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        _metanllb_model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True
        )
    _metanllb_model.eval()
    print(f"MetaNLLB: Loaded model '{_metanllb_model_name}' on device '{device}'")
    
    return _metanllb_model, _metanllb_tokenizer 


def translate_text_metanllb(text: str, source_lang: str, target_lang: str, verbose: bool = False) -> Optional[str]:
    """
    Translate text using Meta's NLLB (No Language Left Behind) local model.

    Args:
        text: Text to translate
        source_lang: Source language code (e.g., 'en') or FLORES-200 code (e.g., 'eng_Latn')
        target_lang: Target language code (e.g., 'fr') or FLORES-200 code (e.g., 'fra_Latn')
        verbose: Print debug information

    Returns:
        Translated text or None on failure
    """
    
       
    global _metanllb_model, _metanllb_tokenizer, _metanllb_model_name
     
    try:
        
        # Map ISO 639-1 codes to FLORES-200 codes if needed
        src_flores = _NLLB_LANG_MAP.get(source_lang, source_lang)
        tgt_flores = _NLLB_LANG_MAP.get(target_lang, target_lang)

        if verbose:
            small_text = (text[:10] + '...') if len(text) > 10 else text
            print(f"MetaNLLB: Translating '{small_text}' from {src_flores} to {tgt_flores} using model '{_metanllb_model_name}'...") 
        
        if _metanllb_model is None or _metanllb_tokenizer is None:
            if verbose:
                print("MetaNLLB: Model or tokenizer not loaded, loading now...")    
            _metanllb_model, _metanllb_tokenizer = _get_metanllb_model(src_flores, tgt_flores, verbose)
            
        if verbose:
            print(f"MetaNLLB: before tokenization, loaded model and tokenizer: '{_metanllb_model_name}'") 

        # _metanllb_tokenizer.src_lang = src_flores # Change source to FLORES-200 code
        # _metanllb_tokenizer.tgt_lang = tgt_flores # Change target to FLORES-200 code
        #_metanllb_tokenizer.return_tensors = 

        inputs = _metanllb_tokenizer(text, return_tensors="pt").to(_metanllb_model.device)  
   
        if verbose:
            print("MetaNLLB: Tokenization complete")
                  
        if verbose:
            input_len = len(inputs['input_ids'][0])
            print(f"MetaNLLB: Input tokens len: {input_len}")   
            
        force_bos_token_id = _metanllb_tokenizer.convert_tokens_to_ids(f"__{tgt_flores}__")
        
        with torch.no_grad():    
            output = _metanllb_model.generate(
            **inputs,
            forced_bos_token_id=force_bos_token_id,
            num_beams=5,    
            max_new_tokens=1000,
            do_sample=False,
            early_stopping=True
           )  
            
        if verbose:
            output_len = len(output[0])
            print(f"MetaNLLB: Output tokens len: {output_len}") 
            
        translated = _metanllb_tokenizer.decode(output[0], skip_special_tokens=True)
        
        if translated is None:
            print("MetaNLLB: FAILED: Decoding returned None")
            return None
    
        if verbose: 
            short_text = (text[:30] + '...') if len(text) > 30 else text
            short_translated = (translated[:30] + '...') if len(translated) > 30 else translated
            print(f"MetaNLLB: '{short_text}' -> '{short_translated}' ({src_flores}->{tgt_flores})")
            
        return translated
    
    except ImportError as e:
        if verbose:
            print(f"MetaNLLB dependencies not installed: {e}")
        return None
    except Exception as exc:
        if verbose:
            print(f"MetaNLLB translation failed: {exc}")
        return None



def _get_marianmt_model(source_lang: Optional[str] = None, target_lang: Optional[str] = None):
    """Load and cache MarianMT model and tokenizer, reloading if the effective model name changed.

    The effective model is ``_marianmt_model_name`` when explicitly set, otherwise
    ``Helsinki-NLP/opus-mt-{source_lang}-{target_lang}`` is auto-derived from the
    language pair provided at translation time.
    """
    global _marianmt_model, _marianmt_tokenizer, _marianmt_loaded_model_name
    if not _TRANSFORMERS_AVAILABLE or not _TORCH_AVAILABLE:
        raise ImportError(
            "MarianMT requires transformers and torch. "
            "Install with: pip install transformers torch"
        )
    if _marianmt_model_name is not None:
        effective_model = _marianmt_model_name
    elif source_lang and target_lang:
        effective_model = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    else:
        raise ValueError(
            "MarianMT: No model name configured and no language pair provided. "
            "Set --marianmt-model or supply source/target language codes."
        )
    if _marianmt_model is None or _marianmt_tokenizer is None or _marianmt_loaded_model_name != effective_model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"MarianMT: Loading tokenizer '{effective_model}' on device '{device}'")
        _marianmt_tokenizer = MarianTokenizer.from_pretrained(effective_model)
        print(f"MarianMT: Loading model '{effective_model}' on device '{device}'")
        _marianmt_model = MarianMTModel.from_pretrained(effective_model)
        _marianmt_model = _marianmt_model.to(device)
        _marianmt_model.eval()
        _marianmt_loaded_model_name = effective_model
        print(f"MarianMT: Loaded model '{effective_model}' on device '{device}'")
    return _marianmt_model, _marianmt_tokenizer


def translate_text_marianmt(text: str, source_lang: str, target_lang: str, verbose: bool = False) -> Optional[str]:
    """
    Translate text using a MarianMT local model (Helsinki-NLP Marian MT).

    The model is selected automatically from the language pair
    (``Helsinki-NLP/opus-mt-{source_lang}-{target_lang}``) unless overridden via
    ``set_marianmt_config()``.

    Args:
        text: Text to translate
        source_lang: Source language code (e.g., 'fr')
        target_lang: Target language code (e.g., 'en')
        verbose: Print debug information

    Returns:
        Translated text or None on failure
    """
    try:
        model, tokenizer = _get_marianmt_model(source_lang, target_lang)
        effective_model = _marianmt_loaded_model_name

        if verbose:
            small_text = (text[:30] + '...') if len(text) > 30 else text
            print(f"MarianMT: Translating '{small_text}' from {source_lang} to {target_lang} using '{effective_model}'...")

        inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            translated_tokens = model.generate(**inputs)

        translated = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        if verbose:
            short_text = (text[:30] + '...') if len(text) > 30 else text
            short_translated = (translated[:30] + '...') if len(translated) > 30 else translated
            print(f"MarianMT: '{short_text}' -> '{short_translated}' ({source_lang}->{target_lang})")

        return translated
    except ImportError as e:
        if verbose:
            print(f"MarianMT dependencies not installed: {e}")
        return None
    except Exception as exc:
        if verbose:
            print(f"MarianMT translation failed: {exc}")
        return None



def translate_text(
    text: str,
    source_lang: str = "en",
    target_lang: str = "es",
    backend: Optional[str] = None,
    verbose: bool = False
) -> Optional[str]:
    """
    Translate text using configured backend.

    Args:
        text: Text to translate
        source_lang: Source language code (default: 'en')
        target_lang: Target language code (default: 'es')
        backend: Backend to use (None = use configured default)
        verbose: Print debug information

    Returns:
        Translated text or None on failure
    """
    if not text or not text.strip():
        return text

    # If source and target are the same, no translation needed
    if source_lang.lower() == target_lang.lower():
        return text

    backend_to_use = backend or _translation_backend

    if backend_to_use == "none" or backend_to_use == "passthrough":
        return text
    elif backend_to_use == "googletrans":
        return translate_text_googletrans(text, source_lang, target_lang, verbose)
    elif backend_to_use == "libretranslate":
        return translate_text_libretranslate(text, source_lang, target_lang, verbose)
    elif backend_to_use == "translategemma":
        return translate_text_translategemma(text, source_lang, target_lang, verbose)
    elif backend_to_use == "metanllb":
        return translate_text_metanllb(text, source_lang, target_lang, verbose)
    elif backend_to_use == "marianmt":
        return translate_text_marianmt(text, source_lang, target_lang, verbose)
    else:
        if verbose:
            print(f"Unknown translation backend: {backend_to_use}")
        return None


__all__ = [
    "translate_text",
    "set_translation_backend",
    "get_translation_backend",
    "set_libretranslate_config",
    "set_translategemma_config",
    "set_metanllb_config",
    "set_marianmt_config",
]
