---
license: apache-2.0
language:
- sa
- en
base_model: google/byt5-small
pipeline_tag: text2text-generation
library_name: transformers
tags:
- sanskrit
- devanagari
- ocr
- post-ocr-correction
- byt5
- indic-nlp
datasets:
- tusharislampure29/sanskrit-ocr-correction
---

# ByT5-Sanskrit-OCR — Post-OCR Correction for Sanskrit/Devanagari

A byte-level **ByT5-small** fine-tuned to correct OCR'd Sanskrit/Devanagari text — dropped vowel
signs, confused nasals (anusvara vs chandrabindu), split conjuncts, visually-similar consonants
(व/ब, श/ष/स), and danda errors (॥ becoming a plain `|`). Trained on synthetic OCR noise from a linguistically-grounded Devanagari corruption
engine. Built on a free Colab T4.

- **Input:** noisy OCR Sanskrit, prefixed with `correct: `
- **Output:** corrected Devanagari Unicode
- **Base:** [`google/byt5-small`](https://huggingface.co/google/byt5-small) (Apache 2.0)
- **Code & report:** https://github.com/tusharislampure29/sanskrit-ocr-correction

## Why byte-level (ByT5)?

mT5's subword tokenizer fragments **89.9% of Sanskrit words** and fragments *more* as the input
degrades. ByT5 is byte-level: **no Devanagari input can produce an `<unk>`** and the decoder can emit
any Unicode sequence — exactly what a corrector of *broken* glyphs needs. The cost is ~3× longer
sequences, acceptable for short verses. (Full analysis in the repo.)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("tusharislampure29/byt5-sanskrit-ocr")
model = AutoModelForSeq2SeqLM.from_pretrained("tusharislampure29/byt5-sanskrit-ocr")

noisy = "कर्मण्येवाधिकारस्त मा फलषु कदाचन |"   # dropped matras + danda-as-pipe
ids = tok("correct: " + noisy, return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_length=256, num_beams=4)[0], skip_special_tokens=True))
```

## Results

Held-out test set (1,800 lines; split by clean line, zero overlap with training), before vs after:

| Metric | OCR output (before) | ByT5-corrected (after) | Δ |
|---|---|---|---|
| WER ↓ | 0.556 | **0.240** | −57% |
| Exact-match ↑ | 0.000 | **0.252** | +0.25 |
| CER ↓ | 0.084 | **0.065** | −22% |

A **strong word-level corrector**: it more than halves the word error rate and returns 25% of lines
exactly correct. CER improves most on **heavily-degraded** input (0.122 → 0.074, −40%); on already-light
noise it can over-correct at the character level (CER 0.047 → 0.057) while still cutting WER (0.363 → 0.192).
Plus a per-error-family **taxonomy** eval (which error types it fixes best) — see the repo.

## Training

ByT5-small, seq2seq, prefix `correct:`, max len 384, lr 3e-4, effective batch 16 (4 × grad-accum 4),
3 epochs, **bf16** on T4 (fp16 is NaN-unstable for T5/ByT5), best-by-validation-CER checkpoint
(val CER 0.033). Dataset: [`tusharislampure29/sanskrit-ocr-correction`](https://huggingface.co/datasets/tusharislampure29/sanskrit-ocr-correction).

## Limitations

Trained on **synthetic** OCR noise — strongest on the modelled error families; novel scanner
artifacts are out-of-distribution. Coverage is classical Sanskrit (Ayurveda/Yoga/Gita/subhashitas).
For production manuscript pipelines, pair with a human-review/abstain step.

## License & citation

Apache 2.0. Built for the ImmverseAI AI/ML assignment and released openly.
Author: Tushar Islampure (github.com/tusharislampure29).
