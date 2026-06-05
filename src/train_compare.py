"""
LoRA vs full fine-tuning — a controlled mini-experiment (assignment bonus).

Trains `google/byt5-small` two ways on the **same data, same budget, same seed**
and reports a fair head-to-head:

  * trainable parameters (count + % of the model)
  * what you actually ship to disk (full checkpoint vs LoRA adapter)
  * training wall-clock (identical optimizer steps)
  * held-out quality — CER / WER / exact-match, before vs after

This is intentionally a *reduced-budget* study (a few hundred steps on a subset),
separate from the production model `tusharislampure29/byt5-sanskrit-ocr`. The point
is the **comparison and the tradeoff**, not a new SOTA number — exactly what the
assignment's bonus asks for.

Why this matters for a corrector: full FT updates all ~300M params (best fit,
~1.1 GB checkpoint); LoRA freezes the base and trains tiny rank-decomposition
adapters (a few MB, swappable per-domain) at some quality cost. Measuring that
cost is the whole experiment.

Run it on a GPU (Colab T4 is plenty) — NOT on a CPU-only laptop, ByT5 full-FT
will crawl. See notebooks/lora_vs_full.ipynb for one-click Run-all.

    python -m src.train_compare --mode both \
        --train data/processed/train.jsonl --test data/processed/test.jsonl \
        --max-train 4000 --max-eval 400 --max-steps 200 \
        --results eval/results/lora_vs_full.json

`--mode full` or `--mode lora` runs just one leg.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .eval_harness import load_jsonl, correct_batch, score_triples

BASE = "google/byt5-small"
PREFIX = "correct: "
MAX_SRC = MAX_TGT = 256

# Each method gets the learning rate it's conventionally run at, so the
# comparison reflects how people actually use them (LoRA needs a higher lr
# because far fewer parameters move). Everything else is held identical.
DEFAULT_LR = {"full": 3e-4, "lora": 1e-3}


def _count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _dir_size_mb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


def _precision_flags():
    """bf16 on capable GPUs (Ampere+), else fp32. NEVER fp16 for T5/ByT5
    (activations overflow fp16 → loss=NaN → dead model)."""
    import torch
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return use_bf16


def _make_tokenized(tok, rows: list[dict]):
    from datasets import Dataset
    ds = Dataset.from_list([{"noisy": r["noisy"], "clean": r["clean"]} for r in rows])

    def preprocess(batch):
        model_in = tok([PREFIX + x for x in batch["noisy"]], max_length=MAX_SRC, truncation=True)
        labels = tok(text_target=batch["clean"], max_length=MAX_TGT, truncation=True)
        model_in["labels"] = labels["input_ids"]
        return model_in

    return ds.map(preprocess, batched=True, remove_columns=ds.column_names)


def train_one(mode: str, train_rows: list[dict], tok, args) -> dict:
    """Train a single leg (full | lora) and return its measured stats."""
    import torch
    from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                              Seq2SeqTrainingArguments, Seq2SeqTrainer)

    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)

    if mode == "lora":
        from peft import LoraConfig, get_peft_model, TaskType
        # T5/ByT5 attention projections are named q / v — the standard LoRA targets.
        lcfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=args.lora_r,
                          lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          target_modules=["q", "v"])
        model = get_peft_model(model, lcfg)

    trainable, total = _count_params(model)
    lr = args.lr if args.lr is not None else DEFAULT_LR[mode]
    use_bf16 = _precision_flags()
    out_dir = f"{args.out}/{mode}"

    targs = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        warmup_ratio=0.05, weight_decay=0.01,
        bf16=use_bf16, fp16=False,            # fp16 NEVER for T5/ByT5
        logging_steps=25, save_strategy="no", report_to="none",
        seed=args.seed,
    )
    collator = DataCollatorForSeq2Seq(tok, model=model)
    tokenized = _make_tokenized(tok, train_rows)
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=tokenized,
                             data_collator=collator)

    t0 = time.perf_counter()
    trainer.train()
    train_secs = round(time.perf_counter() - t0, 1)

    # What you'd actually ship: full model dir vs the LoRA adapter only.
    save_dir = f"{out_dir}/final"
    model.save_pretrained(save_dir)
    ship_mb = _dir_size_mb(save_dir)

    return {
        "mode": mode,
        "learning_rate": lr,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "ship_size_mb": ship_mb,
        "train_seconds": train_secs,
        "max_steps": args.max_steps,
        "precision": "bf16" if use_bf16 else "fp32",
        "_model": model,  # popped before serialisation; used for eval
    }


def evaluate(stat: dict, tok, test_rows: list[dict]) -> dict:
    """Score a trained leg on the held-out test subset (before vs after)."""
    import torch
    model = stat.pop("_model")
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    noisy = [r["noisy"] for r in test_rows]
    preds = correct_batch(tok, model, noisy, max_len=MAX_TGT)
    triples = [{"noisy": r["noisy"], "clean": r["clean"], "pred": p}
               for r, p in zip(test_rows, preds)]
    stat["eval"] = score_triples(triples)
    # free GPU memory between legs
    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stat


def _print_table(results: list[dict]):
    print("\n=== LoRA vs full fine-tuning (same data / budget / seed) ===")
    hdr = f"{'mode':6} {'trainable':>14} {'%':>7} {'ship MB':>9} {'train s':>8} {'CER↓':>14} {'WER↓':>14} {'EM↑':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        e = r["eval"]
        cer = f"{e['cer_before']}→{e['cer_after']}"
        wer = f"{e['wer_before']}→{e['wer_after']}"
        print(f"{r['mode']:6} {r['trainable_params']:>14,} {r['trainable_pct']:>6}% "
              f"{r['ship_size_mb']:>9} {r['train_seconds']:>8} {cer:>14} {wer:>14} "
              f"{e['exact_match_after']:>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lora", "both"], default="both")
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--test", default="data/processed/test.jsonl")
    ap.add_argument("--max-train", type=int, default=4000, help="subset of train rows")
    ap.add_argument("--max-eval", type=int, default=400, help="subset of test rows")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None, help="override; else per-mode default")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="eval/compare_runs")
    ap.add_argument("--results", default="eval/results/lora_vs_full.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)

    train_rows = load_jsonl(args.train)[:args.max_train]
    test_rows = load_jsonl(args.test)[:args.max_eval]
    print(f"train rows: {len(train_rows)}  |  eval rows: {len(test_rows)}  |  "
          f"base: {BASE}  |  steps: {args.max_steps}  seed: {args.seed}")

    modes = ["full", "lora"] if args.mode == "both" else [args.mode]
    results = []
    for m in modes:
        print(f"\n----- training leg: {m} -----")
        stat = train_one(m, train_rows, tok, args)
        stat = evaluate(stat, tok, test_rows)
        results.append(stat)

    _print_table(results)

    payload = {
        "experiment": "lora_vs_full",
        "base_model": BASE,
        "config": {"max_train": len(train_rows), "max_eval": len(test_rows),
                   "max_steps": args.max_steps, "batch_size": args.batch_size,
                   "grad_accum": args.grad_accum, "seed": args.seed,
                   "lora_r": args.lora_r, "lora_alpha": args.lora_alpha},
        "results": results,
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print(f"\nsaved -> {args.results}")


if __name__ == "__main__":
    main()
