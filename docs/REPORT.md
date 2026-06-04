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
lr 3e-4, effective batch 16 (4 × grad-accum 4), 3 epochs, weight decay 0.01, warmup 5%. **bf16 on T4**
— *not* fp16: T5/ByT5 activations overflow fp16's range, sending the loss to `NaN` on step 1 and
producing a dead model (val_loss=nan, CER stuck ≈1.08). bf16's wider exponent (same as fp32) is
numerically safe; on a T4 it runs in software (slower) but correct. `load_best_model_at_end` on
**eval CER** (best checkpoint: val CER 0.033). (LoRA is unnecessary here — ByT5-small full-FT fits a
T4 comfortably; a LoRA-vs-full comparison is the obvious bonus extension.)

## 5. Hardware constraints and optimizations

Free Colab T4 (16 GB), no local GPU. Optimizations: bf16 (fp16 is unsafe for ByT5, see §4); ByT5-**small**
(not base); short sequences; **push-to-Hub every epoch** (`hub_strategy='every_save'`) so a Colab
disconnect resumes in minutes — a lesson carried from a prior project. It paid off directly: the free
T4 was reaped at step 4004/4050 (99%), but the best checkpoint was already on the Hub, so the run was
recovered with **zero retraining** and evaluated locally on CPU. The notebook installs only the
missing pip pieces (upgrading Colab's torch breaks the CUDA pairing).

## 6. Evaluation methodology

CER is the primary metric (right granularity for OCR correction); WER and exact-match are secondary.
Three layers (`src/eval_harness.py`):
1. **Aggregate** before-vs-after on the held-out test split — the uncorrected baseline is the bar.
2. **Per-severity** — light/medium/heavy.
3. **Error taxonomy** — corrupt test lines with *one* family at a time, measure recovery per family.

**Results (2,400-line held-out test set):**

| Metric | before | after | Δ |
|---|---|---|---|
| WER ↓ | 0.554 | **0.239** | −57% |
| Exact-match ↑ | 0.000 | **0.261** | +0.26 |
| CER ↓ | 0.084 | **0.072** | −14% |

Per-severity (CER before → after / WER before → after): light `0.046 → 0.063` / `0.355 → 0.185`;
medium `0.080 → 0.072` / `0.556 → 0.238`; heavy `0.125 → 0.082` / `0.752 → 0.294`.

The model is a **strong word-level corrector at every severity** (WER −48% to −61%, up to 37% of lines
made exactly correct), and a character-level improver on medium/heavy noise. The one regression is CER
on *light* input, where it over-corrects characters even as it fixes whole words (WER still drops 48%) —
analysed in §7. Charts in `eval/charts/`; per-error-family taxonomy in `eval/results/taxonomy.json`.

## 7. Failure cases

Two failure modes show up in the eval, both expected:

1. **Repetition / over-generation on heavy multi-error lines.** When several words are badly mangled,
   the seq2seq decoder occasionally repeats a token instead of recovering the intended word:
   ```
   noisy : आयोुर्वीज्ञान  आयरवद चरक संहिता सुश्रुत स्ंह िता ॥
   model : आयुर्विज्ञान आयुर्विज्ञान आयुर्वदा चरक संहिता सुश्रुत संहिता ।   ← repeats "आयुर्विज्ञान"
   gold  : आयुर्विज्ञान आयुर्वेद चरक संहिता सुश्रुत संहिता ।
   ```
   Here CER actually rises (0.18 → 0.29) — the price of a decoder that will emit anything.
2. **Character-level over-correction on light noise.** On near-clean input the model "fixes" things
   that were already correct, nudging CER up (0.046 → 0.063 on the light split) even though WER still
   improves (0.355 → 0.185). It learned to rewrite aggressively because most training pairs *needed*
   rewriting; a fraction of clean-passthrough pairs, or a confidence/abstain gate (§9), would curb this.

Also expected: consonant-glyph confusions that yield a *valid but wrong* word (no local signal to fix),
and out-of-distribution scanner artifacts absent from the synthetic noise model.

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

### Evaluation appendix
- Qualitative outputs (real held-out predictions):
  ```
  विश्तर््णं तावत् 4१290 कि.मि वर्त ते ॥   →   विस्तीर्णं तावत् ४१२९० कि.मी वर्तते ।   (heavy, CER 0.27→0.00)
  मनोजकुम ारः भारतयःअभ िनता ।           →   मनोजकुमारः भारतीयः अभिनेता ।         (medium, CER 0.18→0.00)
  तेन् देवताः वजयिनः अभबन् ।             →   तेन देवताः विजयिनः अभवन् ।            (light, CER 0.12→0.00)
  ```
- Before vs after: WER 0.554→0.239 (−57%), EM 0.000→0.261, CER 0.084→0.072 (−14%). Charts: `eval/charts/`
  (`eval_comparison.png`, `taxonomy.png`, `training_loss.png`).
- Hallucination/error analysis: see §7 — repetition on heavy lines; character-level over-correction on
  light input.
