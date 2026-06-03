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
    """Stream a HF dataset and auto-detect the Devanagari-bearing text field.

    Default recommendation: `wikimedia/wikipedia:20231101.sa` (Sanskrit Wikipedia,
    Parquet — no loading script, so it works on modern `datasets`). Script-based
    datasets (e.g. rahular/itihasa) are NOT loadable on datasets>=3 without a
    loading script, which is why we don't rely on them.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [hf] `datasets` not installed; skipping HF source.")
        return []
    name, _, config = dataset.partition(":")
    print(f"  [hf] streaming {name}" + (f" (config={config})" if config else ""), flush=True)

    ds = None
    for kw in ({}, {"trust_remote_code": True}):
        try:
            ds = load_dataset(name, config or None, split="train", streaming=True, **kw)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  [hf] load attempt {kw or 'default'} failed: {type(e).__name__}: {e}",
                  flush=True)
    if ds is None:
        print(f"  [hf] giving up on {dataset}; using bundled corpus only.", flush=True)
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
    try:
        for row in ds:
            for s in dev_strings(row):
                out.append(s)
            if len(out) >= max_rows:
                break
    except Exception as e:  # noqa: BLE001
        print(f"  [hf] streaming stopped early: {type(e).__name__}: {e}", flush=True)
    print(f"  [hf] collected {len(out)} Devanagari strings from {name}", flush=True)
    return out


def collect_clean(min_len: int, max_len: int, hf_dataset: str | None,
                  max_hf: int, max_clean: int = 0, seed: int = 13) -> list[str]:
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

    # keep the bundled corpus (it's curated + on-domain) and cap the rest
    if max_clean and len(uniq) > max_clean:
        import random
        bundled = set(clean_line(p) + " ।" for p in load_bundled())  # rough overlap guard
        priority = [c for c in uniq if c in bundled]
        rest = [c for c in uniq if c not in bundled]
        rng = random.Random(seed)
        rng.shuffle(rest)
        uniq = priority + rest[:max(0, max_clean - len(priority))]
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
    ap.add_argument("--max-hf", type=int, default=8000,
                    help="max rows/articles to pull from the HF source")
    ap.add_argument("--max-clean", type=int, default=0,
                    help="cap on total unique clean lines (0 = no cap)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    print("Collecting clean Sanskrit lines...")
    clean = collect_clean(args.min_len, args.max_len, args.hf_dataset, args.max_hf,
                          args.max_clean, args.seed)
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
