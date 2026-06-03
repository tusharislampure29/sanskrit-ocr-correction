"""Tests for the Devanagari noise engine and the evaluation metrics."""
import unicodedata

import pytest

from src.devanagari_noise import (
    corrupt, corrupt_many, OCRNoiseConfig, MATRAS, DANDA, DOUBLE_DANDA,
)
from src.eval_harness import levenshtein, cer, wer, isolated_config, score_triples

GITA = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।"
AYUR = "त्रयो दोषाः वातः पित्तं कफः इति ।"


def test_determinism():
    a = corrupt(GITA, seed=42).noisy
    b = corrupt(GITA, seed=42).noisy
    assert a == b
    # different seed -> (almost always) different corruption
    diff = corrupt(GITA, seed=1).noisy
    assert isinstance(diff, str)


def test_nfc_normalized():
    r = corrupt(AYUR, seed=3)
    assert unicodedata.normalize("NFC", r.clean) == r.clean


def test_events_logged():
    r = corrupt(GITA, config=OCRNoiseConfig().scaled(1.5), seed=5)
    assert r.num_errors == len(r.events)
    if r.num_errors:
        assert all(hasattr(e, "kind") for e in r.events)


def test_corrupt_many_distinct():
    variants = corrupt_many(GITA, 10)
    noisy = [v.noisy for v in variants]
    assert len(noisy) == len(set(noisy))        # all distinct
    assert all(v.noisy != v.clean for v in variants)  # all actually corrupted
    assert len(variants) <= 10


def test_isolated_family_only_that_family():
    # matra_delete only -> every event is a matra_delete, and only matras vanish
    cfg = isolated_config("matra_delete")
    r = corrupt(GITA, config=cfg, seed=0)
    kinds = {e.kind for e in r.events}
    assert kinds.issubset({"matra_delete"})


def test_danda_family_isolated():
    cfg = isolated_config("danda_confuse")
    # corrupt many times; the danda should sometimes change to a known confusable
    seen = set()
    for s in range(20):
        r = corrupt(GITA, config=cfg, seed=s)
        for e in r.events:
            assert e.kind == "danda_confuse"
            seen.add(e.repl)
    assert seen  # at least one danda was confused across the runs


def test_levenshtein():
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("", "abc") == 3
    assert levenshtein(["a", "b"], ["a", "c"]) == 1


def test_cer_wer():
    assert cer("abc", "abc") == 0.0
    assert cer("abd", "abc") == pytest.approx(1/3)
    assert wer("a b c", "a b c") == 0.0
    assert wer("a x c", "a b c") == pytest.approx(1/3)


def test_score_triples_before_after():
    triples = [
        {"noisy": "abx", "clean": "abc", "pred": "abc"},  # perfectly corrected
        {"noisy": "abc", "clean": "abc", "pred": "abc"},  # already clean
    ]
    s = score_triples(triples)
    assert s["cer_after"] == 0.0
    assert s["cer_before"] > 0
    assert s["exact_match_after"] == 1.0


def test_corruption_in_recoverable_range():
    # medium severity over a batch should land in a sane CER band (not destroyed)
    lines = [GITA, AYUR]
    cers = []
    for ln in lines:
        for s in range(10):
            r = corrupt(ln, seed=s)
            cers.append(cer(r.noisy, r.clean))
    avg = sum(cers) / len(cers)
    assert 0.01 < avg < 0.4, f"avg CER {avg} outside recoverable range"
