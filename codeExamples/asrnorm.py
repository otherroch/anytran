import re
from faster_whisper import WhisperModel

class StreamingNormalizer:
    def __init__(self, line_cap=None):
        # Precompile regex for speed
        self.multi_space = re.compile(r'\s+')
        self.space_before_punct = re.compile(r'\s+([,.!?])')
        self.space_after_punct = re.compile(r'([,.!?])([A-Za-z])')
        self.leading_space = re.compile(r'^\s+')
        
        # Common ASR fillers
        self.fillers = re.compile(
            r'\b(um|uh|er|ah|you know|like)\b',
            re.IGNORECASE
        )

        # Quote normalization table
        self.quote_map = str.maketrans({
            "’": "'",
            "`": "'",
            "“": '"',
            "”": '"'
        })

        self.buffer = ""
        self.line_cap = line_cap

    def add_segment(self, text: str):
        """Add a new ASR segment to the buffer and attempt to form sentences."""
        self.buffer += " " + text
        self.buffer = self.normalize_chunk(self.buffer)
        
    def get_sentence(self) -> str:
        """Extract complete sentences from the buffer, respecting line_cap if set."""
        if not self.line_cap:
            # Default behavior: split at sentence-ending punctuation
            sentences = re.split(r'(?<=[.!?])\s+', self.buffer)
            if len(sentences) > 1:
                complete_sentences = sentences[:-1]
                self.buffer = sentences[-1]
                return "\n".join(complete_sentences).strip()
            return ""
        # Smart line breaking with line_cap
        output_lines = []
        buf = self.buffer
        # Updated regex: match punctuation, but ignore commas in numbers and periods in websites
        # We'll use a custom function to filter valid split points
        punct_re = re.compile(r'([,.!?])')
        website_re = re.compile(r'\b(?:www|http|https)\.[\w\.-]+\b', re.IGNORECASE)
        while len(buf) > self.line_cap:
            search_region = buf[:self.line_cap+1]
            matches = list(punct_re.finditer(search_region))
            valid_matches = []
            for m in matches:
                punct = m.group(1)
                idx = m.start()
                # Ignore comma if surrounded by digits (e.g., 300,000)
                if punct == ',' and idx > 0 and idx < len(search_region)-1:
                    if search_region[idx-1].isdigit() and search_region[idx+1].isdigit():
                        continue
                # Ignore period if part of a website (e.g., www.urbanlife.org)
                if punct == '.':
                    # Check if period is part of a website
                    # Find the word containing the period
                    left = max(0, idx-20)
                    right = min(len(search_region), idx+20)
                    snippet = search_region[left:right]
                    if website_re.search(snippet):
                        continue
                valid_matches.append(m)
            if valid_matches:
                last_punct = valid_matches[-1]
                end_idx = last_punct.end()
                output_lines.append(buf[:end_idx].strip())
                buf = buf[end_idx:].lstrip()
            else:
                output_lines.append(buf[:self.line_cap].strip())
                buf = buf[self.line_cap:].lstrip()
        self.buffer = buf
        return "\n".join(output_lines).strip()
    
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

    def finalize(self, text: str):
        """Final cleanup for completed sentences."""
        text = text.strip()
        if not text:
            return ""
        # Capitalize first letter
        text = text[0].upper() + text[1:]
        # Remove extra spaces again
        text = self.multi_space.sub(" ", text)
        return text


import argparse

def main():
    parser = argparse.ArgumentParser(description="ASR Normalization Utility")
    parser.add_argument('--input', required=True, help='Input mp3 file path')
    parser.add_argument('--output', required=True, help='Output text file path')
    parser.add_argument('--normalize', action='store_true', help='Normalize text segments (default: False)')
    parser.add_argument('--line-cap', type=int, default=None, help='Maximum number of characters per output line (only with --normalize)')
    args = parser.parse_args()

    # Determine if input is audio or text file
    is_audio = args.input.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.m4a'))

    if args.normalize:
        normalizer = StreamingNormalizer(line_cap=args.line_cap)
        with open(args.output, 'w', encoding='utf-8') as outfile:
            if is_audio:
                model = WhisperModel("large-v3")
                segments, info = model.transcribe(args.input)
                for segment in segments:
                    normalizer.add_segment(segment.text)
                    sentence = normalizer.get_sentence()
                    if sentence:
                        outfile.write(sentence + '\n')
            else:
                # Treat input as plain text file, normalize each line as a segment
                with open(args.input, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()
                        if not line:
                            continue
                        normalizer.add_segment(line)
                        sentence = normalizer.get_sentence()
                        if sentence:
                            outfile.write(sentence + '\n')
            # Write any remaining buffer as a final sentence
            final = normalizer.finalize(normalizer.buffer)
            if final:
                outfile.write(final + '\n')
    else:
        if is_audio:
            model = WhisperModel("large-v3")
            segments, info = model.transcribe(args.input)
            with open(args.output, 'w', encoding='utf-8') as outfile:
                for segment in segments:
                    outfile.write(segment.text.strip() + '\n')
        else:
            # Just copy text file lines to output
            with open(args.input, 'r', encoding='utf-8') as infile, open(args.output, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    outfile.write(line.strip() + '\n')

if __name__ == "__main__":
    main()