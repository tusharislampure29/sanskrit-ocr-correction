"""
Evaluation harness for Sanskrit post-OCR correction.

Three layers, so we measure the model more than one way and can't fool ourselves:

  1. Aggregate quality   — CER / WER / exact-match on the held-out test set, and
                           the crucial *before vs after*: the uncorrected baseline
                           (noisy vs clean) is the bar the model must beat.
  2. Per-severity        — does the model help on light scans AND degraded ones?
  3. Error taxonomy      — a controlled study: corrupt each test line with ONLY
                           one error family enabled, then measure how much of THAT
                           family the model recovers. Tells you *what* it fixes.

Metrics are computed with a vendored Levenshtein (no heavy deps), at character
level (CER) and whitespace-token level (WER). All strings are NFC-normalized
before scoring so Unicode form differences don't inflate the error rate.

Usage:
  # before numbers (no model needed, CPU):
  py -3.12 -m src.eval_harness --baseline --test data/processed/test.jsonl

  # after numbers (needs torch+transformers; runs on GPU in the notebook):
  py -3.12 -m src.eval_harness --model tusharislampure29/byt5-sanskrit-ocr \
      --test data/processed/test.jsonl --save eval/results/preds_test.json

  # score pre-generated predictions on a CPU-only box (like project 01):
  py -3.12 -m src.eval_harness --load-responses eval/results/preds_test.json

  # controlled per-error-family recovery (needs model):
  py -3.12 -m src.eval_harness --taxonomy --model <id> --test data/processed/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from pathlib import Path

from .devanagari_noise import OCRNoiseConfig, corrupt
import dataclasses


def levenshtein(a, b) -> int:
    """Edit distance over sequences (chars or word lists)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def cer(pred: str, ref: str) -> float:
    pred, ref = nfc(pred), nfc(ref)
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, ref) / len(ref)


def wer(pred: str, ref: str) -> float:
    p, r = nfc(pred).split(), nfc(ref).split()
    if not r:
        return 0.0 if not p else 1.0
    return levenshtein(p, r) / len(r)


def score_triples(triples: list[dict]) -> dict:
    """triples: list of {noisy, clean, pred}. Reports before (noisy) vs after (pred)."""
    base_cer = [cer(t["noisy"], t["clean"]) for t in triples]
    base_wer = [wer(t["noisy"], t["clean"]) for t in triples]
    pred_cer = [cer(t["pred"], t["clean"]) for t in triples]
    pred_wer = [wer(t["pred"], t["clean"]) for t in triples]
    em_before = sum(nfc(t["noisy"]) == nfc(t["clean"]) for t in triples) / len(triples)
    em_after = sum(nfc(t["pred"]) == nfc(t["clean"]) for t in triples) / len(triples)
    b, a = statistics.mean(base_cer), statistics.mean(pred_cer)
    return {
        "n": len(triples),
        "cer_before": round(b, 4),
        "cer_after": round(a, 4),
        "cer_reduction_pct": round((b - a) / b * 100, 1) if b else 0.0,
        "wer_before": round(statistics.mean(base_wer), 4),
        "wer_after": round(statistics.mean(pred_wer), 4),
        "exact_match_before": round(em_before, 4),
        "exact_match_after": round(em_after, 4),
    }


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------- model loading --------------------------------
def load_model(model_id: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tok, model


def correct_batch(tok, model, texts: list[str], max_len: int = 256,
                  prefix: str = "correct: ", batch_size: int = 32) -> list[str]:
    import torch
    preds: list[str] = []
    dev = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        chunk = [prefix + t for t in texts[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_length=max_len, num_beams=4)
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))
    return preds


# ----------------------------- error taxonomy -------------------------------
ERROR_FAMILIES = [
    "matra_confuse", "matra_delete", "anusvara", "visarga_drop",
    "halant_delete", "consonant_confuse", "danda_confuse", "space_delete",
]


def isolated_config(family: str) -> OCRNoiseConfig:
    """A config with every rate 0 except the targeted family (turned up high)."""
    zero = {f.name: 0.0 for f in dataclasses.fields(OCRNoiseConfig)}
    cfg = OCRNoiseConfig(**zero)
    if family == "anusvara":
        cfg.anusvara_delete = 0.8
        cfg.anusvara_confuse = 0.4
    elif family == "matra_confuse":
        cfg.matra_confuse = 0.8
    elif family == "matra_delete":
        cfg.matra_delete = 0.7
    elif family == "consonant_confuse":
        cfg.consonant_confuse = 0.5
    elif family == "halant_delete":
        cfg.halant_delete = 0.7
    elif family == "visarga_drop":
        cfg.visarga_drop = 0.9
    elif family == "danda_confuse":
        cfg.danda_confuse = 0.9
    elif family == "space_delete":
        cfg.space_delete = 0.7
    return cfg


def run_taxonomy(tok, model, clean_lines: list[str]) -> dict:
    report = {}
    for fam in ERROR_FAMILIES:
        cfg = isolated_config(fam)
        triples = []
        for li, c in enumerate(clean_lines):
            r = corrupt(c, config=cfg, seed=li)
            if r.noisy == r.clean:
                continue
            triples.append({"noisy": r.noisy, "clean": r.clean})
        if not triples:
            continue
        preds = correct_batch(tok, model, [t["noisy"] for t in triples])
        for t, p in zip(triples, preds):
            t["pred"] = p
        report[fam] = score_triples(triples)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/processed/test.jsonl")
    ap.add_argument("--model", default=None, help="HF id or local path of the ByT5 model")
    ap.add_argument("--baseline", action="store_true",
                    help="score the uncorrected noisy text (no model)")
    ap.add_argument("--taxonomy", action="store_true",
                    help="controlled per-error-family recovery (needs --model)")
    ap.add_argument("--load-responses", default=None,
                    help="score a saved [{noisy,clean,pred}] file (CPU-only)")
    ap.add_argument("--save", default=None, help="save predictions json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.load_responses:
        triples = json.loads(Path(args.load_responses).read_text(encoding="utf-8"))
        print(json.dumps(score_triples(triples), ensure_ascii=False, indent=2))
        return

    rows = load_jsonl(args.test)
    if args.limit:
        rows = rows[:args.limit]

    if args.baseline:
        # "after" == "before" (identity): shows the bar the model must beat
        triples = [{"noisy": r["noisy"], "clean": r["clean"], "pred": r["noisy"]} for r in rows]
        print("=== BASELINE (uncorrected noisy vs clean) ===")
        print(json.dumps(score_triples(triples), ensure_ascii=False, indent=2))
        # per-severity
        by_sev = {}
        for r in rows:
            by_sev.setdefault(r.get("severity", "all"), []).append(
                {"noisy": r["noisy"], "clean": r["clean"], "pred": r["noisy"]})
        print("--- per severity ---")
        for sev, ts in sorted(by_sev.items()):
            s = score_triples(ts)
            print(f"  {sev}: cer={s['cer_before']}  (n={s['n']})")
        return

    if not args.model:
        raise SystemExit("Provide --model, or use --baseline / --load-responses.")

    tok, model = load_model(args.model)

    if args.taxonomy:
        clean_lines = sorted({r["clean"] for r in rows})
        rep = run_taxonomy(tok, model, clean_lines)
        print("=== ERROR TAXONOMY (isolated family → recovery) ===")
        for fam, s in rep.items():
            print(f"  {fam:18s} cer {s['cer_before']:.3f} -> {s['cer_after']:.3f} "
                  f"({s['cer_reduction_pct']:+.0f}%)  EM {s['exact_match_after']:.2f}")
        Path("eval/results").mkdir(parents=True, exist_ok=True)
        Path("eval/results/taxonomy.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    preds = correct_batch(tok, model, [r["noisy"] for r in rows])
    triples = [{"noisy": r["noisy"], "clean": r["clean"], "pred": p,
                "severity": r.get("severity")} for r, p in zip(rows, preds)]
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(triples, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"saved {len(triples)} predictions -> {args.save}")
    print("=== MODEL (corrected vs clean) ===")
    print(json.dumps(score_triples(triples), ensure_ascii=False, indent=2))
    by_sev = {}
    for t in triples:
        by_sev.setdefault(t.get("severity", "all"), []).append(t)
    print("--- per severity ---")
    for sev, ts in sorted(by_sev.items()):
        s = score_triples(ts)
        print(f"  {sev}: cer {s['cer_before']:.3f} -> {s['cer_after']:.3f} "
              f"({s['cer_reduction_pct']:+.0f}%)")


if __name__ == "__main__":
    main()
