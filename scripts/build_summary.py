"""Assemble eval/results/summary.json from the saved test predictions, and write a
minimal trainer_state.json from the two epochs that were logged on Colab before the
runtime was reaped (so make_charts can render the training curve). Honest record."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval_harness import score_triples

RES = ROOT / "eval" / "results"
triples = json.loads((RES / "preds_test.json").read_text(encoding="utf-8"))
overall = score_triples(triples)

by_sev = {}
for t in triples:
    by_sev.setdefault(t.get("severity", "all"), []).append(t)
per_sev = {sev: score_triples(ts) for sev, ts in sorted(by_sev.items())}

# Per-epoch values observed in the Colab training cell (real logged numbers).
training = {
    "max_steps": 4050,
    "epochs_completed": 2,            # runtime reaped at step 4004/4050; best ckpt = epoch 2
    "precision": "bf16 (fp16 disabled: NaN-unstable for ByT5)",
    "log_history": [
        {"epoch": 1, "step": 1350, "loss": 0.0434, "eval_loss": 0.043056, "eval_cer": 0.034414},
        {"epoch": 2, "step": 2700, "loss": 0.0286, "eval_loss": 0.040927, "eval_cer": 0.033221},
    ],
}

summary = {"test": overall, "per_severity": per_sev, "training": training}
tax = RES / "taxonomy.json"
if tax.exists():
    summary["taxonomy"] = json.loads(tax.read_text(encoding="utf-8"))

(RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

# minimal trainer_state.json for the training-loss chart
state = {"log_history": [
    {"step": s["step"], "loss": s["loss"]} for s in training["log_history"]
] + [
    {"step": s["step"], "eval_loss": s["eval_loss"], "eval_cer": s["eval_cer"]} for s in training["log_history"]
], "max_steps": training["max_steps"]}
(RES / "trainer_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

print("=== OVERALL ===")
print(json.dumps(overall, ensure_ascii=False, indent=2))
print("=== PER SEVERITY (before -> after) ===")
for sev, s in per_sev.items():
    print(f"  {sev:7s} CER {s['cer_before']:.4f} -> {s['cer_after']:.4f}   "
          f"WER {s['wer_before']:.4f} -> {s['wer_after']:.4f}   EM {s['exact_match_after']:.4f}   (n={s['n']})")
print("wrote summary.json + trainer_state.json")
