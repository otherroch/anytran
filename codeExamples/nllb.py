import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RealTimeTranslator:
    def __init__(
        self,
        model_name="facebook/nllb-200-1.3B",
        device="cuda",
        target_lang="fra_Latn"
    ):
        self.device = device
        self.target_lang = target_lang

        print("Loading translation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )

        self.model.eval()

    def set_target_language(self, lang_code):
        self.target_lang = lang_code

    def translate(self, text, source_lang="eng_Latn"):
        if not text.strip():
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                max_length=256
            )

        return self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )