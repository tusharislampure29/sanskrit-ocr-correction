"""
Builder for notebooks/lora_vs_full.ipynb — the LoRA-vs-full mini-experiment
(assignment bonus). Same generate-from-Python pattern as _make_train_nb.py:

    py -3.12 notebooks/_make_compare_nb.py

One-click on a free Colab T4 (Runtime → T4 GPU → Run all). No HF token needed —
this experiment trains two throwaway legs and reports the comparison; it does NOT
touch the published production model.
"""
import json
from pathlib import Path

REPO_URL = "https://github.com/tusharislampure29/sanskrit-ocr-correction"
HF_DATASET_ID = "tusharislampure29/sanskrit-ocr-correction"

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(f"""
# LoRA vs full fine-tuning — Sanskrit Post-OCR Correction (bonus experiment)

A controlled mini-experiment: fine-tune **ByT5-small** two ways — full fine-tuning
and **LoRA** — on the **same data, same budget, same seed**, and compare:

- trainable parameters (count + %)
- what you ship to disk (full checkpoint vs LoRA adapter)
- training wall-clock (identical optimizer steps)
- held-out quality (CER / WER / exact-match, before vs after)

This is a *reduced-budget* study (a subset + a few hundred steps), separate from the
production model `tusharislampure29/byt5-sanskrit-ocr` — the point is the **tradeoff**,
not a new metric. **Run all** on a free Colab T4 (Runtime → Change runtime type → T4 GPU).
No Hugging Face token needed.

Repo: {REPO_URL}
""")

code("""
# 1. GPU check + deps. Colab ships a matched torch — install only the missing pieces.
#    peft is pinned to a version known-good with transformers 4.46.x.
import subprocess, torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
subprocess.run("pip install -q 'transformers==4.46.3' 'peft==0.13.2' 'datasets>=2.20' "
               "'accelerate>=0.34' sentencepiece", shell=True, check=True)
import transformers, peft
print("transformers", transformers.__version__, "| peft", peft.__version__)
""")

code(f"""
# 2. Clone the repo (src/train_compare.py + the curated corpus live here).
import os
if not os.path.exists('sanskrit-ocr-correction'):
    subprocess.run("git clone {REPO_URL}", shell=True, check=True)
else:
    subprocess.run("git -C sanskrit-ocr-correction pull", shell=True, check=True)
os.chdir('sanskrit-ocr-correction')
print(os.getcwd())
""")

code(f"""
# 3. Get data. Pull the SAME published held-out dataset the production model used,
#    so the comparison sits on the real split (no leakage, identical to the model card).
from datasets import load_dataset
import json, os
os.makedirs('data/processed', exist_ok=True)
ds = load_dataset("{HF_DATASET_ID}")
for split, path in [('train','data/processed/train.jsonl'),
                    ('test','data/processed/test.jsonl')]:
    with open(path, 'w', encoding='utf-8') as f:
        for r in ds[split]:
            f.write(json.dumps({{'noisy': r['noisy'], 'clean': r['clean']}}, ensure_ascii=False) + '\\n')
print({{k: len(v) for k, v in ds.items()}})
""")

code("""
# 4. Run BOTH legs in one process (fair: identical data, budget, seed).
#    ~10-15 min on a T4. Tweak --max-steps / --max-train to trade time for signal.
subprocess.run("python -m src.train_compare --mode both "
               "--train data/processed/train.jsonl --test data/processed/test.jsonl "
               "--max-train 4000 --max-eval 400 --max-steps 200 "
               "--results eval/results/lora_vs_full.json", shell=True, check=True)
""")

code("""
# 5. Chart + show the results table.
subprocess.run("python scripts/make_compare_chart.py", shell=True)
import json
data = json.load(open('eval/results/lora_vs_full.json', encoding='utf-8'))
for r in data['results']:
    e = r['eval']
    print(f"{r['mode']:5} | trainable {r['trainable_params']:>12,} ({r['trainable_pct']}%) "
          f"| ship {r['ship_size_mb']:>7} MB | {r['train_seconds']}s "
          f"| CER {e['cer_before']}→{e['cer_after']} | WER {e['wer_before']}→{e['wer_after']} "
          f"| EM {e['exact_match_after']}")
from IPython.display import Image
Image('eval/charts/lora_vs_full.png')
""")

code("""
# 6. (optional) Download artifacts to paste back into the repo / report.
try:
    from google.colab import files
    for f in ['eval/results/lora_vs_full.json', 'eval/charts/lora_vs_full.png']:
        files.download(f)
except Exception as e:
    print('not on Colab or download skipped:', e)
""")

md("""
### Reading the result

LoRA's whole pitch is **parameter efficiency**: it freezes the 300M-param base and
trains tiny rank-decomposition adapters, so you train a fraction of the weights and
ship a few-MB adapter instead of a ~1 GB checkpoint — and you can keep many
per-domain adapters around one base. Full fine-tuning moves every weight, which
usually wins on raw quality for a single task but costs storage and per-task copies.

For a manuscript-correction product you'd weigh that directly: if quality is within a
point or two of full FT, LoRA's swappable few-MB adapters per text-genre (Ayurveda vs
Yoga vs poetry) are the more deployable choice. The numbers above quantify exactly
that tradeoff on this task.
""")

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = Path(__file__).resolve().parent / "lora_vs_full.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, f"({len(CELLS)} cells)")
