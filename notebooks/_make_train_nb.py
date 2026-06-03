"""
Builder for notebooks/train_colab.ipynb.

Keeping the notebook in a generated .ipynb (rather than hand-edited JSON) means
the cells live as readable Python here and the .ipynb is always valid. Run:

    py -3.12 notebooks/_make_train_nb.py

Edit the REPO_URL / HF ids below before the first run.
"""
import json
from pathlib import Path

REPO_URL = "https://github.com/tusharislampure29/sanskrit-ocr-correction"
HF_MODEL_ID = "tusharislampure29/byt5-sanskrit-ocr"
HF_DATASET_ID = "tusharislampure29/sanskrit-ocr-correction"

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(f"""
# Sanskrit Post-OCR Correction — ByT5 fine-tuning (Colab T4)

Trains a byte-level **ByT5-small** to correct OCR'd Sanskrit/Devanagari text, using
a linguistically-grounded synthetic corruption engine. Designed for a **free Colab T4**.

**Before you Run all:**
1. Runtime → Change runtime type → **T4 GPU**.
2. Add your Hugging Face token as a Colab **Secret** named `HF_TOKEN` (🔑 in the left bar),
   with write access. (Used to push checkpoints — so a Colab disconnect costs minutes, not the run.)

Repo: {REPO_URL}  ·  Model → `{HF_MODEL_ID}`  ·  Dataset → `{HF_DATASET_ID}`
""")

code("""
# 1. GPU check + deps. Colab already ships a matched torch/torchvision pair —
#    install ONLY the missing pieces (upgrading torch breaks the CUDA pairing).
import subprocess, torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
subprocess.run("pip install -q 'transformers>=4.44' 'datasets>=2.20' 'accelerate>=0.33' "
               "sentencepiece evaluate huggingface_hub", shell=True, check=True)
""")

code(f"""
# 2. Auth to Hugging Face from the Colab secret.
from huggingface_hub import login
from google.colab import userdata
login(userdata.get('HF_TOKEN'))
HF_MODEL_ID   = "{HF_MODEL_ID}"
HF_DATASET_ID = "{HF_DATASET_ID}"
""")

code(f"""
# 3. Clone the repo (single source of truth for src/ + the curated corpus).
import os
if not os.path.exists('sanskrit-ocr-correction'):
    subprocess.run("git clone {REPO_URL}", shell=True, check=True)
os.chdir('sanskrit-ocr-correction')
print(os.getcwd())
""")

code("""
# 4. Build the dataset on Colab: curated corpus + Ayurveda pages + HF augmentation
#    (rahular/itihasa = clean Sanskrit from the epics). ~30k corrupted->clean pairs.
subprocess.run("python -m src.data_prep --hf-dataset rahular/itihasa "
               "--max-hf 3000 --variants 12", shell=True, check=True)
import json
print(open('data/processed/stats.json', encoding='utf-8').read())
""")

code("""
# 5. Push the dataset to the Hub (so it's a citable artifact + reproducible).
from datasets import load_dataset
ds = load_dataset('json', data_files={
    'train': 'data/processed/train.jsonl',
    'validation': 'data/processed/val.jsonl',
    'test': 'data/processed/test.jsonl'})
print(ds)
try:
    ds.push_to_hub(HF_DATASET_ID, private=False)
    print('dataset pushed ->', HF_DATASET_ID)
except Exception as e:
    print('dataset push skipped:', e)
""")

code("""
# 6. Tokenize. ByT5 is byte-level: no OOV on Devanagari, corruption-invariant vocab.
from transformers import AutoTokenizer
BASE = 'google/byt5-small'
PREFIX = 'correct: '
MAX_SRC = MAX_TGT = 384
tok = AutoTokenizer.from_pretrained(BASE)

def preprocess(batch):
    model_in = tok([PREFIX + x for x in batch['noisy']], max_length=MAX_SRC,
                   truncation=True)
    labels = tok(text_target=batch['clean'], max_length=MAX_TGT, truncation=True)
    model_in['labels'] = labels['input_ids']
    return model_in

tokenized = ds.map(preprocess, batched=True, remove_columns=ds['train'].column_names)
print(tokenized)
""")

code("""
# 7. Metric: character error rate (CER), the right metric for OCR correction.
import numpy as np
def _lev(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[-1]+1, prev[j-1]+(ca!=cb)))
        prev = cur
    return prev[-1]

def cer(pred, ref):
    return _lev(pred, ref)/max(1, len(ref))

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple): preds = preds[0]
    preds = np.where(preds != -100, preds, tok.pad_token_id)
    labels = np.where(labels != -100, labels, tok.pad_token_id)
    dpred = tok.batch_decode(preds, skip_special_tokens=True)
    dref  = tok.batch_decode(labels, skip_special_tokens=True)
    return {'cer': float(np.mean([cer(p, r) for p, r in zip(dpred, dref)]))}
""")

code(f"""
# 8. Train. fp16 on T4. push_to_hub + save each epoch => resume-safe across disconnects.
import transformers
from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
collator = DataCollatorForSeq2Seq(tok, model=model)

# transformers renamed eval_strategy <- evaluation_strategy around v4.41
_evk = 'eval_strategy' if transformers.__version__ >= '4.41' else 'evaluation_strategy'
args_kw = dict(
    output_dir='byt5-sanskrit-ocr', overwrite_output_dir=True,
    learning_rate=5e-4, per_device_train_batch_size=8,
    per_device_eval_batch_size=16, gradient_accumulation_steps=2,
    num_train_epochs=3, warmup_ratio=0.03, weight_decay=0.01,
    fp16=True, predict_with_generate=True, generation_max_length=MAX_TGT,
    logging_steps=50, save_strategy='epoch', save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model='cer', greater_is_better=False,
    push_to_hub=True, hub_model_id=HF_MODEL_ID, hub_strategy='every_save',
    report_to='none')
args_kw[_evk] = 'epoch'
training_args = Seq2SeqTrainingArguments(**args_kw)

trainer = Seq2SeqTrainer(
    model=model, args=training_args,
    train_dataset=tokenized['train'], eval_dataset=tokenized['validation'],
    data_collator=collator, compute_metrics=compute_metrics)
trainer.train()
trainer.push_to_hub()
tok.push_to_hub(HF_MODEL_ID)
print('model pushed ->', HF_MODEL_ID)
""")

code("""
# 9. Evaluate on the held-out TEST set: before (noisy) vs after (corrected).
import json
subprocess.run(f"python -m src.eval_harness --model {HF_MODEL_ID} "
               f"--test data/processed/test.jsonl --save eval/results/preds_test.json",
               shell=True, check=True)
print('--- BASELINE (uncorrected) ---')
subprocess.run("python -m src.eval_harness --baseline --test data/processed/test.jsonl", shell=True)
""")

code("""
# 10. Controlled error-taxonomy: what does the model actually fix?
subprocess.run(f"python -m src.eval_harness --taxonomy --model {HF_MODEL_ID} "
               f"--test data/processed/test.jsonl", shell=True)
""")

code("""
# 11. Qualitative demo on the Ayurveda pages ImmverseAI shipped.
subprocess.run(f"python -m src.inference --model {HF_MODEL_ID} --demo-pages", shell=True)
""")

code("""
# 12. Charts (training loss + before/after CER) and download eval artifacts.
subprocess.run("python scripts/make_charts.py", shell=True)
from google.colab import files
for f in ['eval/results/preds_test.json', 'eval/results/taxonomy.json',
          'eval/charts/eval_comparison.png']:
    try: files.download(f)
    except Exception as e: print('skip', f, e)
""")

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = Path(__file__).resolve().parent / "train_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, f"({len(CELLS)} cells)")
