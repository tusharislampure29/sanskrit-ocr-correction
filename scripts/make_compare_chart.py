"""
Chart for the LoRA-vs-full mini-experiment (assignment bonus).

Reads eval/results/lora_vs_full.json (written by `python -m src.train_compare`)
and renders eval/charts/lora_vs_full.png:

  * left  — trainable parameters, full vs LoRA (log scale): the headline gap.
  * mid   — what you ship to disk, MB (log scale): full checkpoint vs LoRA adapter.
  * right — held-out CER/WER after training (only if the quality run has been done).

Safe to run at any stage: the params/size panels always render (those numbers are
deterministic); the quality panel is drawn only once `eval` is populated.
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

FULL_C, LORA_C = "#D1495B", "#30638E"


def main():
    f = RES / "lora_vs_full.json"
    if not f.exists():
        print("skip lora_vs_full (no lora_vs_full.json yet)")
        return
    data = json.loads(f.read_text(encoding="utf-8"))
    res = {r["mode"]: r for r in data["results"]}
    full, lora = res.get("full"), res.get("lora")
    if not (full and lora):
        print("skip lora_vs_full (need both full and lora legs)")
        return

    have_quality = full.get("eval") and lora.get("eval")
    npanels = 3 if have_quality else 2
    fig, axes = plt.subplots(1, npanels, figsize=(4.6 * npanels, 4.4))

    # panel 1 — trainable params (log)
    ax = axes[0]
    ax.bar(["full", "LoRA"], [full["trainable_params"], lora["trainable_params"]],
           color=[FULL_C, LORA_C])
    ax.set_yscale("log"); ax.set_ylabel("trainable parameters (log)")
    ax.set_title("Trainable parameters")
    for i, r in enumerate([full, lora]):
        ax.text(i, r["trainable_params"], f"{r['trainable_params']:,}\n({r['trainable_pct']}%)",
                ha="center", va="bottom", fontsize=8)

    # panel 2 — ship size (log)
    ax = axes[1]
    ax.bar(["full", "LoRA"], [full["ship_size_mb"], lora["ship_size_mb"]],
           color=[FULL_C, LORA_C])
    ax.set_yscale("log"); ax.set_ylabel("artifact on disk, MB (log)")
    ax.set_title("What you ship")
    for i, r in enumerate([full, lora]):
        ax.text(i, r["ship_size_mb"], f"{r['ship_size_mb']:.1f} MB",
                ha="center", va="bottom", fontsize=8)

    # panel 3 — quality (optional)
    if have_quality:
        ax = axes[2]
        x = range(2); w = 0.35
        fc, lc = full["eval"], lora["eval"]
        ax.bar([i - w/2 for i in x], [fc["cer_after"], fc["wer_after"]], w,
               label="full", color=FULL_C)
        ax.bar([i + w/2 for i in x], [lc["cer_after"], lc["wer_after"]], w,
               label="LoRA", color=LORA_C)
        ax.set_xticks(list(x)); ax.set_xticklabels(["CER↓", "WER↓"])
        ax.set_ylabel("error rate after correction"); ax.set_title("Held-out quality")
        ax.legend()

    cfg = data.get("config", {})
    fig.suptitle(f"LoRA vs full fine-tune — {data['base_model']} "
                 f"(same data/budget/seed; {cfg.get('max_steps','?')} steps)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CHARTS / "lora_vs_full.png", dpi=140)
    print("wrote lora_vs_full.png", "(with quality panel)" if have_quality else "(params/size only)")


if __name__ == "__main__":
    main()
