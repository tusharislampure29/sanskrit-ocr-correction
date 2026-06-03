"""
Generate charts for the README / LinkedIn / report:
  - eval/charts/eval_comparison.png   before vs after CER & WER + exact-match
  - eval/charts/taxonomy.png          per-error-family CER reduction
  - eval/charts/training_loss.png     train/eval loss + eval CER over steps (if logs exist)

Reads artifacts written by the training notebook / eval harness. Skips any chart
whose source data is absent, so it's safe to run at any stage.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"
CHARTS = ROOT / "eval" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)


def _score(triples):
    import sys
    sys.path.insert(0, str(ROOT))
    from src.eval_harness import score_triples
    return score_triples(triples)


def eval_comparison():
    f = RES / "preds_test.json"
    if not f.exists():
        print("skip eval_comparison (no preds_test.json yet)")
        return
    s = _score(json.loads(f.read_text(encoding="utf-8")))
    metrics = ["CER", "WER", "Exact match"]
    before = [s["cer_before"], s["wer_before"], s["exact_match_before"]]
    after = [s["cer_after"], s["wer_after"], s["exact_match_after"]]
    x = range(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - w/2 for i in x], before, w, label="OCR output (before)", color="#D1495B")
    ax.bar([i + w/2 for i in x], after, w, label="ByT5-corrected (after)", color="#30638E")
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylabel("rate (lower is better for CER/WER)")
    ax.set_title(f"Sanskrit OCR correction — CER {s['cer_before']:.3f} → {s['cer_after']:.3f} "
                 f"({s['cer_reduction_pct']:+.0f}%)")
    ax.legend()
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(CHARTS / "eval_comparison.png", dpi=140)
    print("wrote eval_comparison.png")


def taxonomy():
    f = RES / "taxonomy.json"
    if not f.exists():
        print("skip taxonomy (no taxonomy.json yet)")
        return
    rep = json.loads(f.read_text(encoding="utf-8"))
    fams = list(rep.keys())
    before = [rep[k]["cer_before"] for k in fams]
    after = [rep[k]["cer_after"] for k in fams]
    x = range(len(fams)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - w/2 for i in x], before, w, label="before", color="#D1495B")
    ax.bar([i + w/2 for i in x], after, w, label="after", color="#30638E")
    ax.set_xticks(list(x)); ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_ylabel("CER"); ax.set_title("Per-error-family recovery (isolated corruption)")
    ax.legend(); fig.tight_layout(); fig.savefig(CHARTS / "taxonomy.png", dpi=140)
    print("wrote taxonomy.png")


def training_loss():
    # look for a trainer_state.json under the output dir or any checkpoint
    candidates = list(ROOT.glob("byt5-sanskrit-ocr/**/trainer_state.json")) + \
                 list(ROOT.glob("**/trainer_state.json"))
    if not candidates:
        print("skip training_loss (no trainer_state.json yet)")
        return
    state = json.loads(candidates[0].read_text(encoding="utf-8"))
    logs = state.get("log_history", [])
    tr = [(l["step"], l["loss"]) for l in logs if "loss" in l]
    ev = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]
    cer = [(l["step"], l["eval_cer"]) for l in logs if "eval_cer" in l]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if tr: ax.plot(*zip(*tr), label="train loss", color="#999")
    if ev: ax.plot(*zip(*ev), label="eval loss", color="#30638E", marker="o")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(loc="upper right")
    if cer:
        ax2 = ax.twinx()
        ax2.plot(*zip(*cer), label="eval CER", color="#D1495B", marker="s")
        ax2.set_ylabel("eval CER"); ax2.legend(loc="upper center")
    ax.set_title("Training: loss + eval CER")
    fig.tight_layout(); fig.savefig(CHARTS / "training_loss.png", dpi=140)
    print("wrote training_loss.png")


if __name__ == "__main__":
    eval_comparison(); taxonomy(); training_loss()
