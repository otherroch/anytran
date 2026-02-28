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
        # Website regex for normalization (only match www.domain.tld, http(s)://domain.tld, or domain.tld)
        self.website_re = re.compile(
            r'(www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}|https?://[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9\-]+\.[a-zA-Z]{2,})',
            re.IGNORECASE
        )
        # Website whitespace remover function
        self.website_space_remover = lambda match: re.sub(r'\s+', '', match.group(0))

    def add_segment(self, text: str):
        """Add a new ASR segment to the buffer and attempt to form sentences."""
        # For text input, treat each line as a segment, preserve spaces
        if self.buffer:
            self.buffer += '\n'
        # Only normalize websites in the segment, not the whole buffer
        self.buffer += self.website_re.sub(self.website_space_remover, text)

    def get_sentence(self) -> str:
        """Extract complete sentences from the buffer, respecting line_cap if set."""
        # For text input, just return the buffer as lines
        if not self.line_cap:
            lines = self.buffer.split('\n')
            if len(lines) > 1:
                complete_lines = lines[:-1]
                self.buffer = lines[-1]
                return '\n'.join(complete_lines).strip()
            return ''
        # Smart line breaking with line_cap (unchanged)
        output_lines = []
        buf = self.buffer
        punct_re = re.compile(r'([,.!?])')
        website_re = re.compile(r'(w\s*\.?\s*w\s*\.?\s*w\s*\.?\s*[\w\.-]+\s*\.?\s*[a-z]{2,}|https?\s*://\s*[\w\.-]+\s*\.?\s*[a-z]{2,}|[\w\.-]+\s*\.?\s*[a-z]{2,})', re.IGNORECASE)
        while len(buf) > self.line_cap:
            search_region = buf[:self.line_cap+1]
            website_spans = [m.span() for m in website_re.finditer(search_region)]
            for span_start, span_end in website_spans:
                if span_start < self.line_cap and span_end > self.line_cap:
                    extend_len = span_end - len(search_region)
                    if extend_len > 0 and len(buf) > len(search_region):
                        search_region = buf[:len(search_region)+extend_len]
            matches = list(punct_re.finditer(search_region))
            valid_matches = []
            for m in matches:
                punct = m.group(1)
                idx = m.start()
                if punct == ',' and idx > 0 and idx < len(search_region)-1:
                    if search_region[idx-1].isdigit() and search_region[idx+1].isdigit():
                        continue
                inside_website = False
                for span_start, span_end in website_spans:
                    if idx >= span_start and idx < span_end:
                        inside_website = True
                        break
                if inside_website:
                    continue
                valid_matches.append(m)
            def normalize_line(line):
                return website_re.sub(self.website_space_remover, line.strip())
            if valid_matches:
                last_punct = valid_matches[-1]
                end_idx = last_punct.end()
                output_lines.append(normalize_line(buf[:end_idx]))
                buf = buf[end_idx:].lstrip()
            else:
                extended = False
                for span_start, span_end in website_spans:
                    if span_start < self.line_cap and span_end > self.line_cap:
                        output_lines.append(normalize_line(buf[:span_end]))
                        buf = buf[span_end:].lstrip()
                        extended = True
                        break
                if not extended:
                    output_lines.append(normalize_line(buf[:self.line_cap]))
                    buf = buf[self.line_cap:].lstrip()
        self.buffer = website_re.sub(self.website_space_remover, buf)
        output_lines = [website_re.sub(self.website_space_remover, line) for line in output_lines]
        return '\n'.join(output_lines).strip()
        
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
        # Improved website regex: match www., http(s)://, and domain patterns
        # Website regex: match www., http(s)://, and domain patterns, also catch 'www.' alone
        website_re = re.compile(r'(www\.[\w\.-]*|https?://[\w\.-]+\.[a-z]{2,}|[\w\.-]+\.[a-z]{2,})', re.IGNORECASE)
        while len(buf) > self.line_cap:
            search_region = buf[:self.line_cap+1]
            website_spans = [m.span() for m in website_re.finditer(search_region)]
            for span_start, span_end in website_spans:
                if span_start < self.line_cap and span_end > self.line_cap:
                    extend_len = span_end - len(search_region)
                    if extend_len > 0 and len(buf) > len(search_region):
                        search_region = buf[:len(search_region)+extend_len]
            matches = list(punct_re.finditer(search_region))
            valid_matches = []
            for m in matches:
                punct = m.group(1)
                idx = m.start()
                if punct == ',' and idx > 0 and idx < len(search_region)-1:
                    if search_region[idx-1].isdigit() and search_region[idx+1].isdigit():
                        continue
                inside_website = False
                for span_start, span_end in website_spans:
                    if idx >= span_start and idx < span_end:
                        inside_website = True
                        break
                if inside_website:
                    continue
                valid_matches.append(m)
            def normalize_line(line):
                # Remove whitespace from websites/domains in the line
                return self.website_re.sub(self.website_space_remover, self.normalize_chunk(line.strip()))
            if valid_matches:
                last_punct = valid_matches[-1]
                end_idx = last_punct.end()
                output_lines.append(normalize_line(buf[:end_idx]))
                buf = buf[end_idx:].lstrip()
            else:
                extended = False
                for span_start, span_end in website_spans:
                    if span_start < self.line_cap and span_end > self.line_cap:
                        output_lines.append(normalize_line(buf[:span_end]))
                        buf = buf[span_end:].lstrip()
                        extended = True
                        break
                if not extended:
                    output_lines.append(normalize_line(buf[:self.line_cap]))
                    buf = buf[self.line_cap:].lstrip()
        # Normalize any remaining buffer for websites/domains before returning
        self.buffer = self.website_re.sub(self.website_space_remover, buf)
        # Also normalize all output lines for websites/domains
        output_lines = [self.website_re.sub(self.website_space_remover, line) for line in output_lines]
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
        # Remove spaces within website/domain names
        def website_space_remover(match):
            # Remove all whitespace (spaces, tabs, newlines) within website/domain
            return re.sub(r'\s+', '', match.group(0))
        text = self.website_re.sub(website_space_remover, text)
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
        output_lines = []
        if is_audio:
            model = WhisperModel("large-v3")
            segments, info = model.transcribe(args.input)
            for segment in segments:
                normalizer.add_segment(segment.text)
                sentence = normalizer.get_sentence()
                if sentence:
                    output_lines.extend(sentence.split('\n'))
        else:
            # Preserve original input structure, only fix website normalization
            output_lines = []
            with open(args.input, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # Only normalize websites, not the whole chunk
                    line = normalizer.website_re.sub(normalizer.website_space_remover, line)
                    output_lines.append(line.rstrip('\n'))
            # Write output lines exactly as input (with website normalization)
            with open(args.output, 'w', encoding='utf-8') as outfile:
                for line in output_lines:
                    outfile.write(line + '\n')
            return
        # Write any remaining buffer as a final sentence
        final = normalizer.finalize(normalizer.buffer)
        if final:
            output_lines.extend(final.split('\n'))
        # Do NOT apply website normalization to all output lines; normalization is handled per segment.
        with open(args.output, 'w', encoding='utf-8') as outfile:
            for line in output_lines:
                outfile.write(line + '\n')
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