"""
Tokenizer analysis: why a *byte-level* model (ByT5) is the right base for noisy
Sanskrit OCR correction — the project's headline finding.

Two questions:

  1. How badly does a subword tokenizer (mT5's 250k-piece SentencePiece) fragment
     Devanagari vs English? (the OOV/fragmentation problem the assignment PDF flags)

  2. The one that actually decides the architecture: what happens to the tokenizer
     when the input is *corrupted* OCR text? A subword vocabulary is built on clean
     text; corrupted glyphs are rarer → more fragmentation and byte-fallback / <unk>.
     ByT5 is invariant: every UTF-8 byte is always in vocabulary, so corruption can
     never produce an unknown token. For a model whose entire job is to read broken
     Devanagari, that invariance is the whole argument.

Runs on CPU. ByT5 token counts are computed exactly from UTF-8 bytes (ByT5 == bytes
+ a few specials) so they're available even without the library; mT5 needs the
SentencePiece tokenizer (loaded if `transformers` is installed).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .devanagari_noise import corrupt, OCRNoiseConfig

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "tokenizer_analysis"

ENGLISH = [
    "Ayurveda is the traditional system of medicine.",
    "The three doshas govern the balance of the body.",
    "A proper diet increases strength and longevity.",
    "Yoga is the cessation of the fluctuations of the mind.",
    "One should rise early, bathe, and clean the teeth.",
]


def byt5_tokens(s: str) -> int:
    """ByT5 tokenizes to raw UTF-8 bytes (+1 EOS). Exact, library-free."""
    return len(s.encode("utf-8")) + 1


def load_sentence_corpus() -> list[str]:
    lines = []
    f = ROOT / "data" / "seed" / "sanskrit_corpus.txt"
    if f.exists():
        for raw in f.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw and not raw.startswith("#"):
                for part in raw.replace("॥", "।").split("।"):
                    part = part.strip()
                    if len(part) > 8:
                        lines.append(part)
    return lines


def try_load_mt5():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("google/mt5-small")
    except Exception as e:  # noqa: BLE001
        print(f"  [mt5] tokenizer unavailable ({type(e).__name__}); "
              "reporting ByT5-only + qualitative mT5 note.")
        return None


def frag_stats(tok, texts: list[str]) -> dict:
    """mT5 fragmentation: tokens/char, % words split into >=2 pieces, <unk> count."""
    tot_tok = tot_char = split_words = total_words = unk = 0
    unk_id = tok.unk_token_id
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        tot_tok += len(ids)
        tot_char += len(t.replace(" ", ""))
        if unk_id is not None:
            unk += sum(1 for i in ids if i == unk_id)
        for w in t.split():
            total_words += 1
            if len(tok(w, add_special_tokens=False)["input_ids"]) >= 2:
                split_words += 1
    return {
        "tokens_per_char": round(tot_tok / max(1, tot_char), 3),
        "pct_words_fragmented": round(100 * split_words / max(1, total_words), 1),
        "unk_tokens": unk,
        "n_tokens": tot_tok,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    skt = load_sentence_corpus()
    if not skt:
        raise SystemExit("No Sanskrit corpus found.")

    # corrupted copies (medium severity) — the OCR-input regime
    skt_noisy = [corrupt(s, config=OCRNoiseConfig(), seed=i).noisy
                 for i, s in enumerate(skt)]

    report: dict = {"n_sanskrit_lines": len(skt)}

    # --- ByT5: exact, byte-level ---
    bt_clean = statistics.mean(byt5_tokens(s) / max(1, len(s)) for s in skt)
    report["byt5"] = {
        "tokens_per_char_clean": round(bt_clean, 3),
        "tokens_per_char_noisy": round(
            statistics.mean(byt5_tokens(s) / max(1, len(s)) for s in skt_noisy), 3),
        "unk_tokens_clean": 0,
        "unk_tokens_noisy": 0,
        "note": "every UTF-8 byte is in-vocab; corruption can never create an <unk>",
    }

    # --- mT5: subword, needs the tokenizer ---
    mt5 = try_load_mt5()
    if mt5 is not None:
        clean = frag_stats(mt5, skt)
        noisy = frag_stats(mt5, skt_noisy)
        eng = frag_stats(mt5, ENGLISH)
        report["mt5"] = {
            "sanskrit_clean": clean,
            "sanskrit_noisy": noisy,
            "english": eng,
            "frag_ratio_sanskrit_vs_english": round(
                clean["tokens_per_char"] / max(1e-9, eng["tokens_per_char"]), 2),
            "extra_fragmentation_when_noisy_pct": round(
                100 * (noisy["tokens_per_char"] - clean["tokens_per_char"])
                / max(1e-9, clean["tokens_per_char"]), 1),
        }

    OUT.joinpath("stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # human-readable summary
    print("=== Tokenizer analysis ===")
    b = report["byt5"]
    print(f"ByT5 (byte-level): {b['tokens_per_char_clean']} tok/char clean, "
          f"{b['tokens_per_char_noisy']} noisy, 0 <unk> in either.")
    if "mt5" in report:
        m = report["mt5"]
        print(f"mT5 Sanskrit clean: {m['sanskrit_clean']['tokens_per_char']} tok/char, "
              f"{m['sanskrit_clean']['pct_words_fragmented']}% words fragmented, "
              f"{m['sanskrit_clean']['unk_tokens']} <unk>.")
        print(f"mT5 English:        {m['english']['tokens_per_char']} tok/char, "
              f"{m['english']['pct_words_fragmented']}% words fragmented.")
        print(f"=> mT5 is {m['frag_ratio_sanskrit_vs_english']}x less efficient on "
              f"Sanskrit than English.")
        print(f"=> mT5 fragments {m['extra_fragmentation_when_noisy_pct']:+}% MORE "
              f"when the Devanagari is corrupted; ByT5 is unchanged. "
              "This is why we picked ByT5.")
    try:
        make_chart(report)
    except Exception as e:  # noqa: BLE001
        print(f"  (chart skipped: {e})")


def make_chart(report: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "mt5" not in report:
        return
    m = report["mt5"]
    # The real story: how badly the SUBWORD tokenizer fragments words, and that it
    # gets WORSE on corrupted Devanagari. ByT5 (byte-level) has no fragmentation
    # concept and zero OOV — annotated, not bar-compared (different axis).
    cats = ["English\n(mT5)", "Sanskrit\nclean (mT5)", "Sanskrit\nnoisy OCR (mT5)"]
    vals = [m["english"]["pct_words_fragmented"],
            m["sanskrit_clean"]["pct_words_fragmented"],
            m["sanskrit_noisy"]["pct_words_fragmented"]]
    colors = ["#4C9F70", "#E1AD01", "#D1495B"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(cats, vals, color=colors, width=0.6)
    ax.set_ylabel("% of words split into 2+ subword tokens")
    ax.set_ylim(0, 105)
    ax.set_title("Why ByT5 for noisy Sanskrit OCR")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.text(0.98, 0.5,
            "ByT5 (byte-level):\nno subword splitting,\n0 OOV — every byte\nin-vocab, even on\ncorrupted glyphs.",
            transform=ax.transAxes, ha="right", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#30638E", ec="none", alpha=0.85),
            color="white")
    fig.tight_layout()
    fig.savefig(OUT / "fragmentation.png", dpi=140)
    print(f"  chart -> {OUT / 'fragmentation.png'}")


if __name__ == "__main__":
    main()
