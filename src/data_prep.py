"""
Build the Sanskrit post-OCR correction dataset.

Pipeline:
  1. Collect *clean* Sanskrit lines from
       - the bundled curated corpus (data/seed/sanskrit_corpus.txt)
       - the Ayurveda pages ImmverseAI shipped (data/seed/page_*.md)
       - optionally a Hugging Face dataset (--hf-dataset), Devanagari field auto-detected
  2. NFC-normalize, split on danda, length- and script-filter, dedupe.
  3. Split *by clean line* into train/val/test (no clean line crosses splits → no leakage).
  4. For each clean line, synthesize N corrupted variants at mixed severity using
     the linguistically-grounded noise engine. Each pair carries its error log.
  5. Write JSONL ({"noisy","clean","n_errors","severity"}) + a stats summary.

Run (CPU, ~seconds for the bundled corpus):
    py -3.12 -m src.data_prep --variants 30 --out data/processed
    py -3.12 -m src.data_prep --hf-dataset rahular/itihasa --max-hf 8000
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from .devanagari_noise import OCRNoiseConfig, corrupt_many

ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "data" / "seed"
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
DANDA_SPLIT_RE = re.compile(r"[।॥]")


def devanagari_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 0.0
    dev = sum(1 for c in chars if "ऀ" <= c <= "ॿ")
    return dev / len(chars)


def clean_line(s: str) -> str:
    s = unicodedata.normalize("NFC", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_bundled() -> list[str]:
    lines: list[str] = []
    corpus = SEED_DIR / "sanskrit_corpus.txt"
    if corpus.exists():
        for raw in corpus.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw and not raw.startswith("#"):
                lines.append(raw)
    # the Ayurveda pages ImmverseAI provided (ground-truth transcriptions)
    for md in sorted(SEED_DIR.glob("page_*.md")):
        for raw in md.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw and DEVANAGARI_RE.search(raw):
                lines.append(raw)
    return lines


def load_hf(dataset: str, max_rows: int) -> list[str]:
    """Stream a HF dataset and auto-detect the Devanagari-bearing text field."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [hf] `datasets` not installed; skipping HF source.")
        return []
    name, _, config = dataset.partition(":")
    print(f"  [hf] streaming {name}" + (f" ({config})" if config else ""))
    try:
        ds = load_dataset(name, config or None, split="train", streaming=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [hf] could not load {dataset}: {e}")
        return []

    def dev_strings(obj):
        if isinstance(obj, str):
            if devanagari_ratio(obj) > 0.5:
                yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from dev_strings(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from dev_strings(v)

    out: list[str] = []
    for row in ds:
        for s in dev_strings(row):
            out.append(s)
            if len(out) >= max_rows:
                return out
    return out


def collect_clean(min_len: int, max_len: int, hf_dataset: str | None,
                  max_hf: int) -> list[str]:
    raw = load_bundled()
    if hf_dataset:
        raw += load_hf(hf_dataset, max_hf)

    pieces: list[str] = []
    for line in raw:
        for part in DANDA_SPLIT_RE.split(line):
            c = clean_line(part)
            if not c:
                continue
            # re-attach a single danda so the model learns sentence terminators
            c = c + " ।"
            if min_len <= len(c) <= max_len and devanagari_ratio(c) > 0.6:
                pieces.append(c)

    # dedupe, preserve order
    seen: set[str] = set()
    uniq = []
    for c in pieces:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def split_lines(lines: list[str], val_frac: float, test_frac: float, seed: int):
    import random
    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    n = len(idx)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test = [lines[i] for i in idx[:n_test]]
    val = [lines[i] for i in idx[n_test:n_test + n_val]]
    train = [lines[i] for i in idx[n_test + n_val:]]
    return train, val, test


# severity buckets: clean scans, mid-quality, degraded manuscripts
SEVERITIES = {"light": 0.5, "medium": 1.0, "heavy": 1.6}


def make_pairs(lines: list[str], variants: int, base_seed: int) -> list[dict]:
    base_cfg = OCRNoiseConfig()
    rows: list[dict] = []
    per_sev = max(1, variants // len(SEVERITIES))
    for li, clean in enumerate(lines):
        for sev_name, factor in SEVERITIES.items():
            cfg = base_cfg.scaled(factor)
            for r in corrupt_many(clean, per_sev, config=cfg,
                                  base_seed=base_seed + li * 1000):
                rows.append({
                    "noisy": r.noisy,
                    "clean": r.clean,
                    "n_errors": r.num_errors,
                    "severity": sev_name,
                })
    return rows


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--variants", type=int, default=30,
                    help="corrupted variants per clean line (across severities)")
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--hf-dataset", default=None,
                    help="optional HF dataset id, e.g. rahular/itihasa[:config]")
    ap.add_argument("--max-hf", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    print("Collecting clean Sanskrit lines...")
    clean = collect_clean(args.min_len, args.max_len, args.hf_dataset, args.max_hf)
    print(f"  {len(clean)} unique clean lines")
    if len(clean) < 10:
        raise SystemExit("Too few clean lines — check the corpus / HF source.")

    train_c, val_c, test_c = split_lines(clean, args.val_frac, args.test_frac, args.seed)
    print(f"  split (by clean line): train={len(train_c)} val={len(val_c)} test={len(test_c)}")

    out = Path(args.out)
    splits = {
        "train": make_pairs(train_c, args.variants, args.seed),
        "val":   make_pairs(val_c, max(6, args.variants // 3), args.seed + 1),
        "test":  make_pairs(test_c, max(6, args.variants // 3), args.seed + 2),
    }
    for name, rows in splits.items():
        write_jsonl(rows, out / f"{name}.jsonl")
        avg_err = sum(r["n_errors"] for r in rows) / max(1, len(rows))
        print(f"  {name}: {len(rows)} pairs  (avg {avg_err:.1f} errors/pair)")

    stats = {
        "clean_lines": len(clean),
        "splits_lines": {"train": len(train_c), "val": len(val_c), "test": len(test_c)},
        "splits_pairs": {k: len(v) for k, v in splits.items()},
        "variants_per_line": args.variants,
        "severities": SEVERITIES,
        "hf_dataset": args.hf_dataset,
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(f"Wrote dataset + stats.json to {out}")


if __name__ == "__main__":
    main()
