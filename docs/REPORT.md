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

**Results (1,800-line held-out test set; split by clean line, zero overlap with training):**

| Metric | before | after | Δ |
|---|---|---|---|
| WER ↓ | 0.556 | **0.240** | −57% |
| Exact-match ↑ | 0.000 | **0.252** | +0.25 |
| CER ↓ | 0.084 | **0.065** | −22% |

Per-severity (CER before → after / WER before → after): light `0.047 → 0.057` / `0.363 → 0.192`;
medium `0.082 → 0.065` / `0.562 → 0.241`; heavy `0.122 → 0.074` / `0.742 → 0.288`.

The model is a **strong word-level corrector at every severity** (WER −47% to −61%, up to 34% of lines
made exactly correct), and a character-level improver on medium/heavy noise. The one regression is CER
on *light* input, where it over-corrects characters even as it fixes whole words (WER still drops 47%) —
analysed in §7. Charts in `eval/charts/`; per-error-family taxonomy in `eval/results/taxonomy.json`.

## 7. Failure cases

Two failure modes show up in the eval, both expected:

1. **Character-level over-correction on near-clean input.** On lightly-noised text the model "fixes"
   characters that were already correct, nudging CER up (light split `0.047 → 0.057`) even though WER
   still improves (`0.363 → 0.192`). Example — it rewrites a correct भ as म:
   ```
   noisy : भाद्रपदे पत्रादीनां रौगभीः ।
   model : माद्रपदे पत्रादीनां रोगमिः ।   ← भ→म over-correction (CER 0.07 → 0.14)
   gold  : भाद्रपदे पुत्रादीनां रोगभीः ।
   ```
   It learned to rewrite aggressively because most training pairs *needed* rewriting; a fraction of
   clean-passthrough pairs, or a confidence/abstain gate (§9), would curb this.
2. **Truncation / derailment on long lines.** On the longest inputs the decoder sometimes stops early or
   drifts at the tail (`…प्रतीतिर्जायते` → `…प्रार`), a known seq2seq behaviour at the generation-length
   edge — addressable with longer `generation_max_length` and length penalties.

**The taxonomy eval pinpoints the over-correction precisely** (isolated single-family corruption, CER before → after):

| Family the model *recovers* | Δ | | Family it *over-corrects* | Δ |
|---|---|---|---|---|
| consonant_confuse | 0.114 → 0.074 (−35%) | | anusvara | 0.024 → 0.082 |
| halant_delete | 0.085 → 0.064 (−24%) | | visarga_drop | 0.031 → 0.073 |
| space_delete | 0.080 → 0.061 (−24%) | | danda_confuse | 0.021 → 0.049 |
| matra_confuse | 0.092 → 0.072 (−22%) | | (all start < 0.03 CER) | |

The split is sharp and interpretable: the model **genuinely repairs the high-error families** (confused
consonants, split conjuncts, merged words, matras) and **over-corrects families whose inputs were
already nearly clean** (anusvara/visarga/danda all start below 0.03 CER). Note danda still reaches
**53% exact-match** — it changes the right character but sometimes alters a neighbour. This is the
single most useful finding from the eval and the clearest lever for a v2 (selective/abstaining decode).

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
  अस्यां6०0 जणा: परयाणंकरतूं शक्णुवन्ति स्म |   →   अस्यां ६०० जनाः प्रयाणं कर्तुं शक्नुवन्ति स्म ।   (heavy, CER 0.23→0.00)
  पौरातययुरोप दश् स्य अपुेक्षया बूृहत् वर्तते .   →   पौरात्ययुरोपदेशस्य अपेक्षया बृहत् वर्तते ।        (medium, CER 0.19→0.00)
  भगवतः निर्ण यः अन्यथ्ा आसित् I                →   भगवतः निर्णयः अन्यथा आसीत् ।                   (light, CER 0.14→0.00)
  ```
- Before vs after: WER 0.556→0.240 (−57%), EM 0.000→0.252, CER 0.084→0.065 (−22%). Charts: `eval/charts/`
  (`eval_comparison.png`, `taxonomy.png`, `training_loss.png`).
- Hallucination/error analysis: see §7 — character-level over-correction on near-clean input; truncation
  on the longest lines.
