# Sanskrit Post-OCR Correction (ByT5)

> OCR butchers Sanskrit. Scanners drop the small vowel strokes, confuse the nasal marks, split
> conjuncts, and turn the verse-ending double-danda (॥) into a plain `|`. This is a byte-level
> **ByT5** model that *reads broken Devanagari and repairs it* —
> trained entirely on synthetic OCR noise from a linguistically-grounded corruption engine, on a free Colab T4.

**Author** · Tushar Islampure ([github.com/tusharislampure29](https://github.com/tusharislampure29))
**Model** · [`tusharislampure29/byt5-sanskrit-ocr`](https://huggingface.co/tusharislampure29/byt5-sanskrit-ocr) on Hugging Face
**Dataset** · [`tusharislampure29/sanskrit-ocr-correction`](https://huggingface.co/datasets/tusharislampure29/sanskrit-ocr-correction)
**Base** · `google/byt5-small` (Apache 2.0) · **Method** · seq2seq fine-tune, byte-level · **License** · Apache 2.0

[![Open training in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tusharislampure29/sanskrit-ocr-correction/blob/main/notebooks/train_colab.ipynb)
[![Open training in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/tusharislampure29/sanskrit-ocr-correction/blob/main/notebooks/train_colab.ipynb)
&nbsp;·&nbsp; LoRA-vs-full bonus:
[![Open bonus in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tusharislampure29/sanskrit-ocr-correction/blob/main/notebooks/lora_vs_full.ipynb)
[![Open bonus in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/tusharislampure29/sanskrit-ocr-correction/blob/main/notebooks/lora_vs_full.ipynb)
&nbsp;·&nbsp;
![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/license-Apache--2.0-green) ![tests](https://img.shields.io/badge/tests-10%20passing-brightgreen)

> Built for the ImmverseAI AI/ML take-home (Track 3 — Post-OCR Correction for Sanskrit/Indic), and
> shipped as an open, reproducible project. It targets exactly the problem behind BharatiyaGPT:
> turning noisy scans of classical Indian texts (the sample data is **Ayurveda**) into clean,
> searchable Unicode.

---

## TL;DR — the result

**The task:** given an OCR'd Sanskrit line with realistic Devanagari errors, recover the correct text.

**Held-out test set, before vs after (character error rate is the OCR metric that matters):**

| Metric | OCR output (before) | ByT5-corrected (after) | Δ |
|---|---|---|---|
| **WER** ↓ | `0.556` | `0.240` | **−57%** |
| **Exact-match** ↑ | `0.000` | `0.252` | **+0.25** |
| **CER** ↓ | `0.084` | `0.065` | **−22%** |

> Measured on the **1,800-line held-out test split** (`src/eval_harness.py`, byte-level CER, NFC-normalized;
> split **by clean line, zero overlap with training**). The model is a **strong word-level corrector**: it
> more than halves the word error rate and returns **25% of lines exactly correct** (up from zero). Character
> error rate improves most where it matters — on **heavily-degraded** scans (`0.122 → 0.074`, −40%). Honest
> nuance: on already-light noise it can *over-correct* at the character level (CER `0.047 → 0.057`) even while
> it still fixes whole words (WER `0.363 → 0.192`) — see [Limitations](#limitations-honest) and below.

**Per-severity CER (before → after):** light `0.047 → 0.057` · medium `0.082 → 0.065` · heavy `0.122 → 0.074`
**Per-severity WER (before → after):** light `0.363 → 0.192` · medium `0.562 → 0.241` · heavy `0.742 → 0.288`

![before/after CER, WER, exact-match](eval/charts/eval_comparison.png)

---

## The headline engineering decision: why ByT5, not mT5

The assignment explicitly flags Sanskrit's tokenization problem (compound words, rare tokens,
Devanagari + Roman, OOV fragmentation). I measured it (`src/tokenizer_analysis.py`, real numbers):

| Tokenizer | tokens/char | % words fragmented | `<unk>` / OOV behaviour |
|---|---|---|---|
| **mT5** (SentencePiece, 250k) on **English** | 0.31 | 31.0% | rare |
| **mT5** on **Sanskrit (clean)** | 0.49 | **89.9%** | byte-fallback |
| **mT5** on **Sanskrit (corrupted OCR)** | 0.52 | higher | **+5.3% more fragmentation** |
| **ByT5** (byte-level) on Sanskrit | 2.84 | n/a (bytes) | **0 — every byte is in-vocab** |

The decisive point for a *correction* model isn't raw efficiency (mT5 packs more chars per token).
It's **robustness and coverage**:

- mT5 already fragments **~90% of Sanskrit words**, and fragments *more* as the input degrades —
  exactly the regime an OCR-correction model lives in. A subword vocabulary trained on clean text
  is least reliable precisely on the broken glyphs we need to fix.
- ByT5 operates on raw UTF-8 bytes, so **no Devanagari input can ever produce an `<unk>`**, and the
  model can **emit any Unicode sequence** on the output side — essential when the corruption created
  byte patterns the subword vocab never saw.

So I traded ~3× longer sequences (the byte-level tax, mitigated by ByT5-small being only 300M and
lines being short verses) for **guaranteed coverage of arbitrary noisy Devanagari**. That trade is
the whole reason the model works. ![tokenizer chart](eval/tokenizer_analysis/fragmentation.png)

## The centerpiece: a linguistically-grounded Devanagari OCR-noise engine

There's almost no labelled "OCR-error → correct" Sanskrit data, so the data *is* the project.
Instead of random character flips, `src/devanagari_noise.py` models **10 error families that mirror
how OCR actually fails on Devanagari**, each calibrated to real failure modes:

| Family | Example | Why it's real |
|---|---|---|
| matra confuse | ि↔ी, े↔ै, ो↔ौ | short/long vowel hooks look near-identical |
| matra delete | कारस्ते → कारस्त | the small stroke is the #1 missed mark |
| anusvara/nasal | पित्तं → पित्त / पित्तँ | anusvara vs chandrabindu, easily swapped or dropped |
| visarga loss | दोषाः → दोषा / दोषा: | dot-pair dropped or read as ASCII `:` |
| halant/virama | क्त → कत | conjuncts split when the halant is missed |
| consonant glyph | व↔ब, घ↔ध, भ↔म, श↔ष↔स | visually confusable letters |
| **danda** | ॥ → `|` | the exact error in the assignment PDF's own example |
| word boundary | फलेषु कदाचन → फलेषुकदाचन | Devanagari OCR mis-segments words |
| unicode/nukta | NFC → NFD, क़ → क+़ | normalization noise |
| digit swap | ३० → 30 | Devanagari ↔ ASCII digits |

It's deterministic per seed, logs **every error it injects** (so evaluation can measure recovery
*per family*), runs at 3 severity levels, and round-trips through Unicode NFC. The Bhagavad Gita
2.47 verse from the assignment example is in the demo. Zero heavy dependencies — it runs in CI.

## Evaluation, three ways (so I can't fool myself)

1. **Aggregate** — CER / WER / exact-match on the held-out test split, always reported **before vs after**.
2. **Per-severity** — does it help on clean scans *and* degraded manuscripts?
3. **Error taxonomy** — a controlled study: corrupt each test line with **only one error family
   enabled**, then measure how much of *that* family the model recovers. This is what tells you
   *what the model actually fixes* (e.g. is it great at matra restoration but weak on consonant
   confusion?) — far more diagnostic than a single CER number. (`--taxonomy`)

Split is **by clean line**, so no source verse leaks between train/val/test.

## How it was built

1. **Data** (`src/data_prep.py`) — curated public-domain corpus (Bhagavad Gita, Patanjali Yoga
   Sutras, Charaka-style Ayurveda, subhashitas — matching ImmverseAI's IKS domains) + the 5 Ayurveda
   pages from the assignment + Hugging Face augmentation from **Sanskrit Wikipedia**
   (`wikimedia/wikipedia:20231101.sa`, Parquet — loads reliably on modern `datasets`) for scale.
   Split by clean line → corrupt at 3 severities → `{noisy, clean}` JSONL.
2. **Train** (`notebooks/train_colab.ipynb`) — ByT5-small, seq2seq, prefix `correct:`, **bf16** on T4
   (fp16 is NaN-unstable for T5/ByT5 — activations overflow → dead model; bf16's wider exponent is
   safe), 3 epochs, `load_best_model_at_end` on eval-CER, **push to HF Hub every epoch** so a Colab
   disconnect costs minutes (a hard-won lesson from project 01 — and it paid off here when the runtime
   was reaped at 99%; the best checkpoint was already on the Hub).
3. **Eval** (`src/eval_harness.py`) — the three layers above, plus a qualitative demo on the
   Ayurveda pages.

## Reproduce

```powershell
git clone https://github.com/tusharislampure29/sanskrit-ocr-correction
cd sanskrit-ocr-correction
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt      # CPU: data prep, tokenizer analysis, scoring

python -m src.data_prep --variants 30                 # quick local build (bundled corpus only)
# the exact dataset the model trained on (Sanskrit Wikipedia augmentation, deterministic split):
python -m src.data_prep --hf-dataset "wikimedia/wikipedia:20231101.sa" \
    --max-hf 4000 --max-clean 4000 --variants 12
python -m src.eval_harness --baseline                 # the "before" numbers
python -m src.tokenizer_analysis                       # ByT5 vs mT5 (real numbers + chart)
python -m pytest -q                                    # 10 tests

# Train on a free Colab T4: open notebooks/train_colab.ipynb, set T4 runtime,
# add HF_TOKEN to Colab Secrets, Run all. It builds the (augmented) dataset, trains,
# pushes the model + dataset to the Hub, evaluates, and runs the taxonomy + demo.

# Bonus — LoRA vs full fine-tuning (no token needed): open notebooks/lora_vs_full.ipynb
# on a T4, Run all. Trains both ways on identical data/budget/seed and reports the tradeoff.
```

Scoring is decoupled from a GPU (like project 01): the notebook saves predictions JSON, and
`python -m src.eval_harness --load-responses eval/results/preds_test.json` scores them on any CPU.

## Sample I/O

Real held-out test predictions (noisy OCR → ByT5 correction → gold):

```
# heavy noise — ASCII digits, split conjuncts, dropped matras, danda-as-pipe  (CER 0.23 → 0.00)
noisy : अस्यां6०0 जणा: परयाणंकरतूं शक्णुवन्ति स्म |
model : अस्यां ६०० जनाः प्रयाणं कर्तुं शक्नुवन्ति स्म ।
gold  : अस्यां ६०० जनाः प्रयाणं कर्तुं शक्नुवन्ति स्म ।

# medium noise — merged words + dropped matras + ASCII period for danda  (CER 0.19 → 0.00)
noisy : पौरातययुरोप दश् स्य अपुेक्षया बूृहत् वर्तते .
model : पौरात्ययुरोपदेशस्य अपेक्षया बृहत् वर्तते ।
gold  : पौरात्ययुरोपदेशस्य अपेक्षया बृहत् वर्तते ।

# light noise — split word, matra slips, ASCII 'I' for danda  (CER 0.14 → 0.00)
noisy : भगवतः निर्ण यः अन्यथ्ा आसित् I
model : भगवतः निर्णयः अन्यथा आसीत् ।
gold  : भगवतः निर्णयः अन्यथा आसीत् ।

# honest failure — on near-clean input the model over-corrects a correct letter (भ→म)  (CER 0.07 → 0.14)
noisy : भाद्रपदे पत्रादीनां रौगभीः ।
model : माद्रपदे पत्रादीनां रोगमिः ।
gold  : भाद्रपदे पुत्रादीनां रोगभीः ।
```

## Bonus — LoRA vs full fine-tuning

A controlled mini-experiment (`src/train_compare.py`, `notebooks/lora_vs_full.ipynb`): fine-tune
ByT5-small **two ways on identical data, budget, and seed**, and compare. The parameter/storage gap
is deterministic and measured directly:

| | Full fine-tune | LoRA (r=16) | Ratio |
|---|---|---|---|
| Trainable params | 299,637,760 (100%) | **1,187,840 (0.40%)** | **252× fewer** |
| Artifact shipped | ~1,143 MB checkpoint | **4.55 MB adapter** | **~250× smaller** |

![LoRA vs full](eval/charts/lora_vs_full.png)

And the quality, measured on Kaggle GPU at an **equal 200-step budget** (held-out 400 lines):

| | Full fine-tune | LoRA (r=16) |
|---|---|---|
| WER ↓ | 0.582 → **0.372** (−36%) | 0.582 → 0.519 (−11%) |
| Exact-match ↑ | **0.10** | 0.033 |
| train time (P100) | 320 s | 248 s (−23%) |

**Honest read:** at an equal, deliberately small budget, **full fine-tuning wins on quality** — it
learns word-level correction much faster (WER −36% vs −11%, 3× the exact-match). LoRA's win is pure
**efficiency**: 252× fewer trainable params, a 250× smaller artifact, ~23% faster. (Both are
*under-trained* at 200 steps — neither improves CER yet; the production model trains far longer and
*does* cut CER. This is a budget-controlled tradeoff study, not the production run.) The deployment
takeaway: full FT for best single-task quality; LoRA when you need many swappable few-MB per-domain
adapters and can afford more steps to close the gap. Reproducible: `notebooks/lora_vs_full.ipynb`.

## The gap this closes (vs prior work)

Sanskrit post-OCR correction has been done before, and I cite it because that's the honest landscape:
[Maheshwari et al. (EMNLP 2022)](https://aclanthology.org/2022.findings-emnlp.466/) and
[Nehrdich et al., ByT5-Sanskrit (2024)](https://arxiv.org/abs/2409.13920) both built byte-level
Sanskrit correctors. **But both report the task as a single accuracy number on a fixed dataset** — you
can't see *which* error types the model actually fixes, and you can't controllably stress-test it.
Prior synthetic-data work for Devanagari is empirical and opaque too (RoundTripOCR re-OCRs rendered
fonts; Guan & Greene use CV glyph-similarity). **That's the gap I went after.**

**What I built that they didn't: a *diagnosable* corrector.** An explicit, linguistically-grounded
Devanagari corruption engine that generates any of 10 **named error families** on demand, wired 1:1 to
a **per-error-family recovery evaluation**. So the result isn't "it scores X" — it's "it reliably fixes
dropped vowel signs, split conjuncts and merged words, and still struggles with these exact cases."
That controllability and interpretability is what a real manuscript pipeline needs before it can trust
a corrector — and it's what the prior open work left on the table. (Full citations in the report.)

## What I'd do with more time

- Train on **real** scanned-manuscript OCR output (Tesseract/Google Vision on GRETIL scans), not
  only synthetic noise, and measure the synthetic-to-real transfer gap.
- A **confidence/abstain** signal so the corrector flags lines it isn't sure about for human review
  (the realistic deployment shape for a manuscript-digitization pipeline).
- Romanized (IAST/HK) ↔ Devanagari transliteration errors as an 11th noise family.
- Distill to a smaller/quantized model for on-device correction.

## Limitations (honest)

- The model is trained on **synthetic** noise. It will be strongest on the error families it was
  shown; truly novel scanner artifacts are out of distribution. The taxonomy eval is there precisely
  to expose where it's weak.
- ByT5's byte-level sequences are ~3× longer than subword — fine for short verses, a cost for long
  passages (chunk them).
- Coverage is classical Sanskrit (IKS domains); modern/technical Sanskrit is under-represented.

## Acknowledgements

GRETIL & the Digital Corpus of Sanskrit for public-domain texts · Sanskrit Wikipedia
(`wikimedia/wikipedia:20231101.sa`) for clean Sanskrit at scale · Google for ByT5 (Apache 2.0) ·
ImmverseAI for the problem framing and the Ayurveda pages.

## Contact

[`@tusharislampure29`](https://github.com/tusharislampure29) · tusharislampure@gmail.com
