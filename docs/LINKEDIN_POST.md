# LinkedIn post

Primary draft below (recruiter hook on top, technical depth in the middle, honest limits at the
bottom). Numbers marked `‹after›` get filled from the Colab eval before posting. **User clicks Post.**

---

## Primary draft (recommended)

> OCR systems quietly butcher Sanskrit. They drop the tiny vowel strokes, confuse ं and ँ, split
> conjuncts, and turn the verse-ending ॥ into a plain `|`. That broken text then poisons every
> downstream step — search, translation, RAG — which is the real bottleneck for digitizing India's
> manuscript heritage.
>
> So this week I built a model that reads broken Devanagari and repairs it. On a free Colab T4.
>
> Model:   huggingface.co/tusharislampure29/byt5-sanskrit-ocr
> Dataset: huggingface.co/datasets/tusharislampure29/sanskrit-ocr-correction
> Code:    github.com/tusharislampure29/sanskrit-ocr-correction
>
> Two decisions did the heavy lifting:
>
> **1. Byte-level over subword.** I measured it: Google's mT5 tokenizer fragments 89.9% of Sanskrit
> words (vs 31% of English) — and fragments *even more* as the input degrades, which is exactly the
> regime an OCR-correction model lives in. ByT5 works on raw UTF-8 bytes, so no Devanagari input can
> ever produce an unknown token and the model can emit any Unicode it needs. I paid ~3× longer
> sequences for guaranteed coverage of arbitrary broken glyphs. For a corrector, that trade is the
> whole game.
>
> **2. The data is the project.** There's almost no labelled "OCR-error → correct" Sanskrit, so I
> wrote a linguistically-grounded Devanagari corruption engine — 10 error families that mirror how
> OCR actually fails (matra deletion, ं/ँ confusion, halant-drop splitting conjuncts, व↔ब glyph
> confusion, ॥→`|`, word-boundary merges, Unicode/nukta noise), not random character flips. It logs
> every error it injects, so I can evaluate recovery *per error family* — not just one aggregate
> number.
>
> Results on a held-out test set (character error rate, the metric that matters for OCR):
> ‹CER 0.086 → ‹after›, a ‹XX›% reduction; plus a per-error-family taxonomy of what it fixes best›
>
> Honest about what it is: trained on synthetic noise, so it's strongest on the error families it
> was shown — novel scanner artifacts are out of distribution (which is exactly why I built the
> taxonomy eval to expose where it's weak). For a real manuscript pipeline you'd pair it with a
> human-review/abstain step.
>
> This was a take-home for an Indian-knowledge-systems AI team, but it's the kind of unglamorous
> infrastructure that decides whether a digitization product actually works. If you work on Indic
> NLP or document AI, I'd love to compare notes. Apache 2.0, full eval + engineering report in the repo.
>
> #IndicNLP #Sanskrit #OCR #NLP #MachineLearning #OpenSource #ByT5 #DocumentAI

## Image attachments (in order)

1. `eval/tokenizer_analysis/fragmentation.png` — the mT5-vs-ByT5 fragmentation chart (the visual hook).
2. `eval/charts/eval_comparison.png` — before/after CER, WER, exact-match.
3. `eval/charts/taxonomy.png` — per-error-family recovery.

## Posting checklist

- [ ] Fill `‹after›` numbers from `eval/results/preds_test.json` (cross-check the README table).
- [ ] HF model card + README updated with the same numbers.
- [ ] Three charts generated (`python scripts/make_charts.py`) and attached in order.
- [ ] Marathi/Devanagari renders correctly in the preview; links show previews.
- [ ] **User clicks Post.** Claude does not click Post.
