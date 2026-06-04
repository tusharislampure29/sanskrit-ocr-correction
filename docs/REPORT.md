# Sanskrit Post-OCR Correction — Technical Report

**Author:** Tushar Islampure · **Track:** Option 3 (Post-OCR Correction for Sanskrit/Indic) ·
**Compute:** free Google Colab T4 · **Code:** github.com/tusharislampure29/sanskrit-ocr-correction

This report follows the structure the assignment requested. Numbers marked `‹after›` are filled by
the training run (`notebooks/train_colab.ipynb`); everything else is measured and in the repo.

## 1. Problem understanding

OCR on Sanskrit/Devanagari fails in characteristic ways: dropped vowel signs (matras), confused
nasals (ं/ँ), split conjuncts when the halant (्) is missed, visually-confusable consonants (व/ब,
श/ष/स), and danda errors (॥ read as `|`). These break downstream NLP — search, translation, RAG —
which is exactly the bottleneck for a manuscript-digitization product like BharatiyaGPT. The task:
a model that takes noisy OCR output and returns clean, correctly-encoded Sanskrit. I framed it as
**monolingual seq2seq denoising** (`noisy → clean`), the standard and most controllable post-OCR
correction setup.

## 2. Dataset preparation

There is little labelled "OCR-error → correct" Sanskrit, so I **generated** the supervision signal
with a linguistically-grounded corruption engine (`src/devanagari_noise.py`) over clean text:

- **Clean sources:** a curated public-domain corpus spanning ImmverseAI's IKS domains — Bhagavad
  Gita, Patanjali Yoga Sutras, Charaka-style Ayurveda, classical subhashitas — plus the 5 Ayurveda
  pages shipped with the assignment, plus augmentation from **Sanskrit Wikipedia**
  (`wikimedia/wikipedia:20231101.sa`, Parquet — loads reliably on modern `datasets`, unlike
  script-based sets) for scale.
- **Pipeline:** NFC-normalize → split on danda → length/script filter → dedupe → **split by clean
  line** (so no source verse leaks across train/val/test) → corrupt each line into multiple variants
  at 3 severity levels (`light`/`medium`/`heavy`).
- **Noise model (the core idea):** 10 error families, each mirroring a real Devanagari OCR failure
  (matra confuse/delete/insert, anusvara, visarga, halant, consonant-glyph confusion, danda,
  word-boundary, unicode/nukta, digit). Deterministic per seed; every injected error is logged so
  evaluation can measure recovery per family.

Output: `{noisy, clean, n_errors, severity}` JSONL, pushed to the Hub as a citable dataset.

## 3. Why this base model (ByT5)

I compared tokenizers directly (`src/tokenizer_analysis.py`, real numbers):

- mT5's SentencePiece fragments **89.9% of Sanskrit words** (vs 31% English; 1.6× more tokens/char)
  and fragments **+5.3% more on corrupted input**.
- ByT5 is **byte-level**: every UTF-8 byte is in-vocab, so **no Devanagari input can produce an
  `<unk>`** and the decoder can emit any Unicode sequence.

For a corrector that must read *broken* glyphs and output *exact* Unicode, byte-level coverage beats
subword efficiency. I accepted ~3× longer sequences (mitigated by short verses + ByT5-small's 300M
size) for that guarantee. This is the central model-understanding tradeoff of the project.

## 4. Fine-tuning approach

Seq2seq fine-tune of `google/byt5-small`, prefix `"correct: "`, source/target max length 384,
lr 5e-4, effective batch 16 (8 × grad-accum 2), 3 epochs, weight decay 0.01, warmup 3%. fp16 on T4.
`load_best_model_at_end` on **eval CER**. (LoRA is unnecessary here — ByT5-small full-FT fits a T4
comfortably; full-FT is simpler and the model is small. A LoRA-vs-full comparison is the obvious
bonus extension.)

## 5. Hardware constraints and optimizations

Free Colab T4 (16 GB), no local GPU. Optimizations: fp16; ByT5-**small** (not base); short
sequences; **push-to-Hub every epoch** (`hub_strategy='every_save'`) so a Colab disconnect resumes
in minutes — a lesson carried from a prior project where a long run was lost to a session reap. The
notebook installs only the missing pip pieces (upgrading Colab's torch breaks the CUDA pairing).

## 6. Evaluation methodology

CER is the primary metric (right granularity for OCR correction); WER and exact-match are secondary.
Three layers (`src/eval_harness.py`):
1. **Aggregate** before-vs-after on the held-out test split — the uncorrected baseline is the bar.
2. **Per-severity** — light/medium/heavy.
3. **Error taxonomy** — corrupt test lines with *one* family at a time, measure recovery per family.

Baseline (measured, bundled corpus): **CER 0.086**, WER 0.454, exact-match 0.000;
per-severity CER 0.048 / 0.079 / 0.131. Post-training: `‹after numbers + taxonomy table›`.

## 7. Failure cases

Expected (and what the taxonomy eval is designed to surface): heavy multi-error lines; consonant
glyph confusions that produce a *valid* but wrong word (no local signal to fix); word-boundary
errors on rare compounds; and out-of-distribution scanner artifacts not in the synthetic model.
`‹concrete failure examples from the eval run›`

## 8. Challenges encountered

- **No ground-truth OCR-error corpus** → had to build the noise model; getting it *linguistically
  realistic* (not random flips) was the real work.
- **Tokenizer choice** drove the whole design; quantifying it (mT5 fragmentation, ByT5 invariance)
  turned a hunch into a decision.
- **Leakage risk:** naive pair-level splitting would put corrupted copies of the same verse in both
  train and test → split by clean line instead.
- **Unicode subtlety:** Devanagari has precomposed vs decomposed forms; everything is NFC-normalized
  before both corruption and scoring so CER reflects real errors, not normalization noise.

## 9. What I'd improve with more time

Train on **real** Tesseract/Vision OCR of scanned GRETIL manuscripts and measure synthetic→real
transfer; add a **confidence/abstain** output for human-in-the-loop manuscript pipelines; add
IAST/Harvard-Kyoto transliteration errors as an 11th family; **LoRA vs full-FT** and a quantized
on-device variant (the assignment's bonus items); a small Gradio demo.

---

### Evaluation appendix (filled by the training run)
- Qualitative outputs: `‹noisy → corrected pairs›`
- Before vs after: `‹table›` · Metrics: CER/WER/EM · Charts: `eval/charts/`
- Hallucination/error analysis: `‹where the model over-corrects or invents text›`
