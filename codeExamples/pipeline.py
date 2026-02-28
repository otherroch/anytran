from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/translategemma-12b-it",
    device="cuda",
    dtype=torch.bfloat16
)

# ---- Text Translation ----
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": "cs",
                "target_lang_code": "de-DE",
                "text": "V nejhorším případě i k prasknutí čočky.",
            }
        ],
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])

# ---- Text Extraction and Translation ----
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source_lang_code": "cs",
                "target_lang_code": "de-DE",
                "url": "https://c7.alamy.com/comp/2YAX36N/traffic-signs-in-czech-republic-pedestrian-zone-2YAX36N.jpg",
            },
        ],
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
