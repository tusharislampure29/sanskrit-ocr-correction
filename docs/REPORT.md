# Sanskrit Post-OCR Correction — Technical Report

**Author:** Tushar Islampure · **Track:** Option 3 (Post-OCR Correction for Sanskrit/Indic) ·
**Compute:** free Google Colab T4 · **Code:** github.com/tusharislampure29/sanskrit-ocr-correction

This report follows the structure the assignment requested. Numbers marked `‹after›` are filled by
the training run (`notebooks/train_colab.ipynb`); everything else is measured and in the repo.

## 1. Problem understanding

OCR on Sanskrit/Devanagari fails in characteristic ways: dropped vowel signs (matras), confused
nasals (anusvara vs chandrabindu), split conjuncts when the halant/virama is missed, visually-confusable consonants (व/ब,
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
**eval CER** (best checkpoint: val CER 0.033). The production model is full fine-tuned (ByT5-small fits a
T4 comfortably); I also ran a controlled **LoRA-vs-full** comparison as a bonus — see §10.

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
IAST/Harvard-Kyoto transliteration errors as an 11th family; a **quantized** on-device variant; a
small Gradio demo. (The **LoRA-vs-full** bonus item is now implemented — see §10.)

## 10. Bonus — LoRA vs full fine-tuning

A controlled mini-experiment (`src/train_compare.py`, `notebooks/lora_vs_full.ipynb`): fine-tune
`google/byt5-small` two ways — full FT and **LoRA** (r=16, α=32, on the `q`/`v` attention
projections) — on the **same data, same budget, same seed**, and compare parameter efficiency,
storage footprint, and held-out quality. It's a deliberately reduced-budget study (a 4k-line subset,
200 steps) separate from the production model; the point is the **tradeoff**, not a new metric.

The parameter and storage gap is deterministic and measured directly:

| | Full fine-tune | LoRA (r=16) | Ratio |
|---|---|---|---|
| Trainable params | 299,637,760 (100%) | **1,187,840 (0.40%)** | **252× fewer** |
| Artifact shipped | ~1,143 MB (fp32 checkpoint) | **4.55 MB adapter** | **~250× smaller** |

That is LoRA's value proposition, quantified: you train 0.4% of the weights and ship a few-MB
adapter instead of a ~1 GB checkpoint — and you can keep many per-domain adapters around one frozen
base. **Held-out quality** (run on Kaggle GPU, equal 200-step budget, 400 test lines):

| | Full fine-tune | LoRA (r=16) |
|---|---|---|
| WER ↓ | 0.582 → **0.372** (−36%) | 0.582 → 0.519 (−11%) |
| Exact-match ↑ | **0.10** | 0.033 |
| CER | 0.087 → 0.096 | 0.087 → 0.145 |
| train time (P100) | 320 s | 248 s |

**Honest interpretation.** At an equal, deliberately small budget, **full fine-tuning clearly wins
on quality** — it learns word-level correction much faster (WER −36% vs −11%, 3× the exact-match).
LoRA's advantage is purely **efficiency** (252× fewer params, 250× smaller artifact, ~23% faster
per the same step count). Both are *under-trained* at 200 steps, so neither improves CER yet — note
the production model (trained far longer) *does* cut CER by 22%; this section is a budget-controlled
tradeoff study, not the production run. **Product read:** full FT for best single-task quality; LoRA
when you need many swappable few-MB per-domain adapters (Ayurveda / Yoga / poetry) and can spend more
steps to close the quality gap. The numbers make that tradeoff concrete instead of hand-wavy.
Artifacts: `eval/results/lora_vs_full.json`, `eval/charts/lora_vs_full.png`.

## 11. Related work & honest positioning

I want to be precise about what here is novel and what isn't — overclaiming in front of an NLP
team is worse than honest scoping.

**What is *not* novel.** Post-OCR correction is a mature task with two ICDAR shared tasks
(2017/2019). Byte-level **ByT5 for post-OCR correction is a near-standard recipe** across many
languages (Icelandic, Swedish, English, Dutch). And **Sanskrit-specific, open, byte-level post-OCR
correction already exists**:
- **Maheshwari et al., "A Benchmark and Dataset for Post-OCR Text Correction in Sanskrit"**
  (Findings of EMNLP 2022) — the canonical Sanskrit post-OCR benchmark on *real* OCR of 30 scanned
  books; their best model is **ByT5 + SLP1**. [aclanthology.org/2022.findings-emnlp.466](https://aclanthology.org/2022.findings-emnlp.466/) · [code](https://github.com/ayushbits/pe-ocr-sanskrit)
- **Nehrdich et al., "ByT5-Sanskrit"** (2024) — an Apache-2.0 byte-level model that sets SOTA on
  Sanskrit OCR post-correction (among other tasks). [arXiv:2409.13920](https://arxiv.org/abs/2409.13920)

So "byte-level ByT5 for Sanskrit post-OCR" is **established prior art, not a first**. This project
re-derives that design choice independently and explains *why* it's right (the tokenizer analysis).

**What is the actual contribution (and it's an engineering/eval one, not a new model).** Prior
Devanagari synthetic-data work generates noise *empirically* — **RoundTripOCR** (Kashid &
Bhattacharyya, ICON 2024) renders fonts and re-OCRs them; **Guan & Greene** (2024) use CV
glyph-similarity feature matching. I found **no prior work that builds an explicit, hand-authored,
linguistically-grounded Devanagari corruption engine organized into named error families**
(matra/anusvara/visarga/halant/consonant-glyph/danda/word-boundary/unicode/digit) **with a matching
per-error-family recovery taxonomy**. That combination — *interpretable, controllable, rule-based
error injection tied 1:1 to a per-family CER/WER breakdown*, packaged end-to-end as an open,
reproducible HF model + dataset — is the uncommon and defensible piece. The numbers above
(CER/WER on a held-out split) are competitive-table-stakes; the **diagnostic taxonomy and the
controllable data engine** are what this project adds to the open ecosystem.

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
