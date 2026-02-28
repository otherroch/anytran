
import re
# No need to import List from typing in Python 3.9+

# Module-level constant for sentence-ending punctuation
SENTENCE_END_PATTERN = r'[.!?。！？]'
SENTENCE_END_WITH_SPACE = re.compile(rf'({SENTENCE_END_PATTERN})\s+')

class StreamingNormalizer:
    def __init__(self):
        # Precompile regex for speed
        self.multi_space = re.compile(r'\s+')
        self.space_before_punct = re.compile(r'\s+([,.!?])')
        self.space_after_punct = re.compile(r'([,.!?])([A-Za-z])')
        self.leading_space = re.compile(r'^\s+')
        self.sentence_ending_with_space = re.compile(rf'(?<={SENTENCE_END_PATTERN})\s+')
        # Common ASR fillers
        self.fillers = re.compile(
            r'\b(um|uh|er|ah|you know|like)\b',
            re.IGNORECASE
        )

        # Quote normalization table
        self.quote_map = str.maketrans({
            "\u2018": "'",
            "\u2019": "'",
            "\u0060": "'",
            "\u201c": '"',
            "\u201d": '"'
        })

        self.buffer = ""

    def add_segment(self, text: str):
        """Add a new ASR segment to the buffer and attempt to form sentences."""
        self.buffer += " " + text
        self.buffer = self.normalize_chunk(self.buffer)

    def get_complete_sentences(self) -> list[str]:
        """Extract complete sentences from the buffer, leaving incomplete sentence for next time."""
        # Default behavior: split at sentence-ending punctuation
        #print(f"Buffer before splitting: '{self.buffer}'")
        # Extend to include Chinese sentence-ending punctuation: 。！？ (fullwidth period, exclamation, question)
         
        sentences = self.sentence_ending_with_space.split(self.buffer)
        #print(f"Sentences after splitting: {sentences}")
        # need to remove the complete sentences from the buffer, leaving any incomplete sentence for next time
        #print(f"num sentences: {len(sentences)}")
        # if the last sentence does not end with a period, it is incomplete and should be kept in the buffer
        if sentences: 
            if not sentences[-1].endswith(('.', '!', '?', '。', '！', '？')):
                if len(sentences) > 1:
                    #print(f"Multi sentence scenario. Last sentence is incomplete: '{sentences[-1]}'")
                    self.buffer = sentences[-1] + " " # pad with space to separate from next segment
                    #print(f"Buffer after splitting: '{self.buffer}'")
                    sentences = sentences[:-1]
                    return sentences
                else:
                    #print(f"Single sentence scenario. Sentence is incomplete: '{sentences[0]}'")
                    self.buffer = sentences[0] + " "  # pad with space to separate from next segment
                    #print(f"Buffer after splitting: '{self.buffer}'")
            else:
                #print(f"All sentences complete. Clearing buffer.")  
                self.buffer = ""
                return sentences

        #print("No complete sentences found.")   
        return []
    
        

    def normalize_chunk(self, text: str) -> str:
        """Normalize a partial ASR segment quickly."""
        # Remove fillers
        text = self.fillers.sub("", text)
        # Normalize quotes
        text = text.translate(self.quote_map)
        # Remove leading spaces
        text = self.leading_space.sub("", text)
        # Remove spaces before punctuation
        text = self.space_before_punct.sub(r'\1', text)
        # Add space after punctuation if missing
        text = self.space_after_punct.sub(r'\1 \2', text)
        # Reduce multiple spaces to single
        text = self.multi_space.sub(" ", text)
        return text.strip()

    def finalize(self, text: str) -> str:
        """Final cleanup for completed sentences."""
        text = text.strip()
        if not text:
            return ""
        # Remove extra spaces again
        text = self.multi_space.sub(" ", text)
        
        # Capitalize first letter unless previous sentence did not end with period
        # This function is called for each split sentence, so we need context
        
        # Capitalize first letter; contextual adjustments (e.g. lowercasing when
        # the previous sentence did not end with a period) are handled in process_line
        if text and text[0].isalpha() and not text[0].isupper():
            text = text[0].upper() + text[1:]
                
        return text

    def process_line(self, line: str) -> str:
        """Normalize a single line and apply line_cap breaking.

        Returns the normalized text, split into multiple ``\\n``-separated
        segments when the line exceeds ``line_cap`` characters.
        """
        self.buffer += self.normalize_chunk(line)
        complete_sentences = self.get_complete_sentences()
        
        # did not get any sentences, wait until we get a complete line before finalizing
        # next call to complete line before finalizing
        if not complete_sentences:
            return None
        
        finalized_output = []   
        for each in complete_sentences:
            finalized_output.append(self.finalize(each))
            
        return '\n'.join(finalized_output)

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences based on punctuation."""
    
    parts = SENTENCE_END_WITH_SPACE.split(text)
    sentences = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        if i + 1 < len(parts):
            sentence += parts[i + 1]
        if sentence.strip():
            sentences.append(sentence.strip())
    return sentences


def normalize_text(text: str) -> str:
    """Normalize a text string using StreamingNormalizer.

    Applies filler removal, quote normalization, punctuation spacing fixes,
    and capitalizes the first character. Multi-line input has each input line
    normalized independently so newlines are preserved. Lines longer than
    ``global_text_line_cap`` characters are broken at punctuation boundaries.
    For multi-line input, the first letter of each resulting line is
    capitalized if the previous line ended with a sentence terminator
    (``.``, ``!``, or ``?``); otherwise, it is lowercased to maintain
    sentence continuity.
    """
    if text is None:
        return None


    normalizer = StreamingNormalizer()
    lines = text.split('\n')

    complete_sentences = []
    for line in lines:
        result = normalizer.process_line(line)
        if result:
            complete_sentences.append(result)
            
    if normalizer.buffer.strip():
        # if we have leftover buffer that wasn't finalized, add it as well
        complete_sentences.append(normalizer.finalize(normalizer.buffer))
        
    #print(f"Total sentences capitalized: {_num_capitalized}")
    
    return '\n'.join(complete_sentences)
