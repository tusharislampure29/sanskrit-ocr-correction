"""
Inference: load the fine-tuned ByT5 corrector and clean up noisy Sanskrit OCR text.

    py -3.12 -m src.inference --model tusharislampure29/byt5-sanskrit-ocr \
        --text "कर्मण्येवाधिकारस्त मा फलषु कदाचन |"

    # or correct the Ayurveda demo pages ImmverseAI shipped:
    py -3.12 -m src.inference --model <id> --demo-pages
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "correct: "


class Corrector:
    def __init__(self, model_id: str, device: str | None = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def correct(self, text: str, max_len: int = 256, num_beams: int = 4) -> str:
        import torch
        enc = self.tok(PREFIX + text, return_tensors="pt", truncation=True,
                       max_length=max_len).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_length=max_len, num_beams=num_beams)
        return self.tok.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--demo-pages", action="store_true",
                    help="run on the Ayurveda page transcriptions in data/seed")
    args = ap.parse_args()

    c = Corrector(args.model)
    if args.text:
        print("IN :", args.text)
        print("OUT:", c.correct(args.text))
    if args.demo_pages:
        from .devanagari_noise import corrupt
        for md in sorted((ROOT / "data" / "seed").glob("page_*.md")):
            for line in md.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                noisy = corrupt(line, seed=7).noisy
                print(f"\nclean : {line}\nnoisy : {noisy}\ncorrect: {c.correct(noisy)}")


if __name__ == "__main__":
    main()
