from anytran.normalizer import StreamingNormalizer, normalize_text

text = "um hello uh world. this is a test,  \n"
#text += "this is only a test. \n"
#text += "some sentences may be incomplete, for example,  this one is incomplete\n"
#text += "and this one, but the normalizer should handle it.  "
print("Original text:")
print(text)
normalized = normalize_text(text)
print("Normalized text:")
print(normalized)


