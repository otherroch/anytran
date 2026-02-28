# Loop Translation: `--looptran` and `--tran-converge`

This document explains the iterative back-translation features available for text file input.

---

## Overview

When translating a text file, it is sometimes useful to translate the output back into the original language repeatedly. This technique—commonly called *back-translation*—helps assess translation quality and stability: a good translation tends to converge toward a stable, consistent form after a few round-trips.

voicetran supports this workflow through two options:

- **`--looptran N`**: Repeat the translation N additional times, alternating language direction.
- **`--tran-converge`**: Automatically stop looping when the output has stabilized (converged).

---

## `--looptran <N>`

### Requirements

- `--input` must be a text file (`.txt`). Audio input files are not supported.
- `--slate-text` must be specified (output of each pass is saved here).
- `--input-lang` and `--output-lang` must be different languages.

### How It Works

Given `--input notes.txt --input-lang fr --output-lang en --slate-text slate.txt --looptran 3`:

| Pass | Input file | Input lang | Output lang | Output file |
|------|-----------|------------|-------------|-------------|
| 0 (initial) | `notes.txt` | `fr` | `en` | `slate.txt` |
| 1 | `slate.txt` | `en` | `fr` | `slate_1.txt` |
| 2 | `slate_1.txt` | `fr` | `en` | `slate_2.txt` |
| 3 | `slate_2.txt` | `en` | `fr` | `slate_3.txt` |

Each pass alternates the translation direction. Odd-numbered output files are in the original source language; even-numbered output files are in the target language.

### Basic Example

Translate a French text to English, then back-translate 4 times:

```bash
anytran --input french_notes.txt \
  --input-lang fr \
  --output-lang en \
  --slate-text slate.txt \
  --slate-backend googletrans \
  --looptran 4
```

This produces: `slate.txt`, `slate_1.txt`, `slate_2.txt`, `slate_3.txt`, `slate_4.txt`.

### Using a Local Translation Backend

For privacy or offline use, combine with a local backend:

```bash
anytran --input article.txt \
  --input-lang de \
  --output-lang en \
  --slate-text translated.txt \
  --slate-backend marianmt \
  --looptran 6
```

### With Scribe Output

If `--scribe-text` is also specified, each loop pass creates a corresponding scribe file:

```bash
anytran --input notes.txt \
  --input-lang es \
  --output-lang en \
  --slate-text slate.txt \
  --scribe-text scribe.txt \
  --looptran 3
```

Produces: `scribe.txt`, `scribe_1.txt`, `scribe_2.txt`, `scribe_3.txt` (and the corresponding slate files).

---

## `--tran-converge`

### Overview

When running many loop iterations, translations often stabilize after just a few passes. The `--tran-converge` flag enables automatic early stopping: after each iteration, the output is compared to the output from **two iterations back** (i.e., the same language direction). If they are identical, convergence is detected and looping stops.

### Convergence Check Logic

For iteration `i` (starting at 1), the check compares:
- `slate_<i>.txt` with `slate_<i-2>.txt` (or `slate.txt` when `i=2`)

Both files represent translations in the same language. If they match exactly, the translation has stabilized.

### Example

```bash
anytran --input document.txt \
  --input-lang fr \
  --output-lang en \
  --slate-text slate.txt \
  --slate-backend googletrans \
  --looptran 20 \
  --tran-converge
```

Output might be:
```
[tran-converge] Convergence detected at iteration 4: 'slate_4.txt' matches 'slate_2.txt'. Stopping early.
```

Instead of running all 20 iterations, the loop stopped at iteration 4 because the translation had stabilized.

### Convergence Example Walk-through

| Iteration | Output file | Content | Convergence check | Converged? |
|-----------|-------------|---------|-------------------|-----------|
| initial | `slate.txt` | "Hello World" | — | — |
| 1 | `slate_1.txt` | "Bonjour le Monde" | — | — |
| 2 | `slate_2.txt` | "Hello World" | `slate_2` vs `slate.txt` | No (differ) |
| 3 | `slate_3.txt` | "Bonjour le Monde" | `slate_3` vs `slate_1` | No (differ) |
| 4 | `slate_4.txt` | "Hello World" | `slate_4` vs `slate_2` | **Yes** → stop |

### Tips

- Use `--looptran` with a high number (e.g., 20) and rely on `--tran-converge` to stop early.
- Most translations converge within 4–8 iterations for typical content.
- If no convergence is reached, all N iterations complete normally.
- Convergence comparison is **exact** (byte-for-byte file comparison). Minor formatting differences prevent detection.

---

## Common Use Cases

### Quality Verification

Translate content back and forth to verify round-trip quality:

```bash
anytran --input report.txt \
  --input-lang en \
  --output-lang de \
  --slate-text german.txt \
  --looptran 4 \
  --tran-converge \
  --slate-backend libretranslate \
  --libretranslate-url http://localhost:5000
```

Compare `german.txt` and `german_2.txt` (both English→German) to see if the translation stabilized.

### Translation Stability Testing

Compare successive same-language outputs to measure stability:

```bash
anytran --input speech.txt \
  --input-lang en \
  --output-lang fr \
  --slate-text fr_output.txt \
  --looptran 6 \
  --slate-backend metanllb
```

Manually compare `fr_output.txt`, `fr_output_2.txt`, and `fr_output_4.txt` to see if the French output stabilizes.

---

## Limitations

- Only works with `--input` text files (`.txt`). Audio input ignores `--looptran`.
- Each pass runs the full translation pipeline, so processing time scales with N.
- `--tran-converge` uses exact file comparison; normalized output (whitespace, punctuation differences) can prevent convergence detection. Use `--no-norm` if exact comparison is critical.
- The loop only runs if `--input-lang ≠ --output-lang`, `--slate-text` is set, and both language codes are specified.

---

## See Also

- [Command Line Options Reference](OPTIONS.md) — full list of all options
- [Text Translation Documentation](TEXT_TRANSLATION.md) — translation backends
