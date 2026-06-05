# LinkedIn post

Plain, human, first-person. No bare combining Devanagari marks (they render as broken
dotted-circle placeholders on most platforms) — marks are described by name instead.
Fits LinkedIn's 3,000-char limit. **User clicks Post.**

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
> raw bytes, so nothing is ever an "unknown token" and it can output any character it needs. You pay
> for that with longer sequences, but for a corrector it's worth it.
>
> 2. The data was the real work. There's almost no labelled "noisy OCR to clean text" data for
> Sanskrit, so I wrote a corruption engine that breaks clean verses the way OCR actually breaks them:
> dropped vowel signs, confused nasal marks, split conjuncts, look-alike consonant swaps, verse marks
> turned into pipes, merged words. Ten error types in all, and it logs every error it injects, so I can
> measure how well the model recovers each type instead of hiding behind one average number.
>
> Results on an 1,800-line held-out set: word error rate went from 0.56 to 0.24, and a quarter of the
> lines came out exactly correct (up from zero). It helps most on the worst, most degraded lines.
>
> I also ran a quick LoRA vs full fine-tuning comparison on the same budget. Full fine-tuning corrects
> better; LoRA trains a tiny slice of the weights and ships as a 4.5 MB file instead of about 1 GB.
> Good to know which one to reach for.
>
> To be straight about it: this is not the first Sanskrit OCR-correction model. There's solid prior
> work (Maheshwari et al. 2022, and the ByT5-Sanskrit model from 2024). What I think is actually useful
> here is that the data generation is fully transparent and controllable, and the evaluation tells you
> which error types the model fixes and which it still misses.
>
> It's all open under Apache 2.0, with the full write-up in the repo. If you work on Indic NLP or
> document AI, I'd genuinely like to hear how you'd approach it.
>
> #Sanskrit #IndicNLP #OCR #NLP #MachineLearning #OpenSource

## Optional image attachments

LinkedIn auto-generates a GitHub link preview (shows the repo + a chart thumbnail), which is enough.
If you'd rather attach images, use the Photo button and pick, in order:
1. `eval/tokenizer_analysis/fragmentation.png` — mT5-vs-ByT5 fragmentation (the visual hook).
2. `eval/charts/eval_comparison.png` — before/after CER, WER, exact-match.
3. `eval/charts/taxonomy.png` — per-error-family recovery.

## Posting checklist

- [x] Written in a plain first-person voice; no bare combining marks that render broken.
- [x] Numbers match the repo (WER 0.56 → 0.24, 25% exact); honest about prior art.
- [x] Fits the 3,000-char limit.
- [ ] **User clicks Post.** (Optionally attach the charts first.)
