def translate_batch(self, texts):
    inputs = self.tokenizer(
        texts,
        padding=True,
        return_tensors="pt"
    ).to(self.device)

    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang]
        )

    return self.tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )