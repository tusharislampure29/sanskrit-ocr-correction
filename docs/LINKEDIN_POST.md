# LinkedIn post

Plain, human, first-person. No bare combining Devanagari marks (they render as broken
dotted-circle placeholders off-platform) — marks are named instead. Leads with the genuine
gap this work closes (diagnosability), while honestly acknowledging prior art. Fits the
3,000-char limit. **User clicks Post.**

---

## Primary draft

> A small frustration that turned into this week's project.
>
> OCR on old Sanskrit manuscripts usually reads the consonants fine, then mangles the little marks
> that actually carry the meaning. It drops a vowel sign, puts a nasal mark in the wrong place, splits
> a conjunct because the joining stroke went missing, or reads the double-danda that ends a verse (॥)
> as a plain pipe. The text ends up looking almost right but being subtly wrong, and that quietly
> poisons everything downstream: search, translation, RAG.
>
> So I built a small model that reads this broken Devanagari and repairs it. It runs on a free Colab T4.
>
> Model: huggingface.co/tusharislampure29/byt5-sanskrit-ocr
> Code: github.com/tusharislampure29/sanskrit-ocr-correction
>
> Two choices mattered more than the model itself.
>
> 1. Byte-level instead of subword. I measured it before committing: mT5's tokenizer splits 89.9% of
> Sanskrit words into fragments (for English it's 31%), and it fragments even more as the text gets
> noisier, which is exactly the situation an OCR corrector works in. A byte-level model (ByT5) reads
> raw bytes, so nothing is ever an "unknown token" and it can output any character it needs.
>
> 2. The data was the real work. There's almost no labelled "noisy OCR to clean text" data for
> Sanskrit, so I wrote a corruption engine that breaks clean verses the way OCR actually breaks them:
> dropped vowel signs, confused nasal marks, split conjuncts, look-alike consonant swaps, verse marks
> turned into pipes, merged words. Ten error types, and it logs every error it injects.
>
> Results on an 1,800-line held-out set: word error rate went from 0.56 to 0.24, and a quarter of the
> lines came out exactly correct (up from zero). It helps most on the worst, most degraded lines.
>
> Here's where I think I moved the needle. Sanskrit OCR-correction has been attempted before
> (Maheshwari et al. 2022, the ByT5-Sanskrit model in 2024), but it's been reported as a single
> accuracy score on a fixed dataset. The piece that was missing is the piece I went after: making the
> whole thing diagnosable. My corruption engine can generate any specific error type on demand, and
> the evaluation measures recovery for each type separately. So instead of "the model scores X", you
> get "it reliably fixes dropped vowel signs, split conjuncts and merged words, and still struggles
> with these exact cases" — which is what you actually need before trusting a corrector on real
> manuscripts.
>
> I also compared LoRA vs full fine-tuning: full fine-tuning corrects better, while LoRA trains a tiny
> slice of the weights and ships as a 4.5 MB file instead of ~1 GB.
>
> It's all open under Apache 2.0, with the full write-up in the repo. If you work on Indic NLP or
> document AI, I'd like to hear how you'd approach it.
>
> #Sanskrit #IndicNLP #OCR #NLP #MachineLearning #OpenSource

## Optional image attachments

LinkedIn auto-generates a GitHub link preview (repo + a chart thumbnail), which is enough. To attach
images instead, use the Photo button and pick, in order:
1. `eval/tokenizer_analysis/fragmentation.png` — mT5-vs-ByT5 fragmentation (the visual hook).
2. `eval/charts/eval_comparison.png` — before/after CER, WER, exact-match.
3. `eval/charts/taxonomy.png` — per-error-family recovery.

## Posting checklist

- [x] Plain first-person voice; no bare combining marks that render broken.
- [x] Leads with the real gap closed (diagnosability) while honestly citing prior art.
- [x] Numbers match the repo; fits the 3,000-char limit.
- [ ] **User clicks Post.** (Optionally attach the charts first.)
