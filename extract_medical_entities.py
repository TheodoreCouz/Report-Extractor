"""Extract medical entities from a French medical report with CamemBERT-bio-gliner.

Usage:
    python extract_medical_entities.py report.txt
    python extract_medical_entities.py report.txt --json
    python extract_medical_entities.py report.txt --labels Maladie Médicament --threshold 0.4
    cat report.txt | python extract_medical_entities.py -
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from gliner import GLiNER

DEFAULT_MODEL = Path(__file__).parent / "models" / "camembert-bio-gliner-v0.1"

DEFAULT_LABELS = [
    "Maladie",
    "Symptôme",
    "Médicament",
    "Dosage",
    "Examen",
    "Procédure",
    "Anatomie",
    "Résultat d'examen",
]

# The model was trained with max_len=384 words; keep a margin for label tokens.
MAX_WORDS_PER_CHUNK = 250

_WORD_RE = re.compile(r"\w+(?:[-_']\w+)*|\S")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _split_spans(text: str) -> list[tuple[int, int]]:
    """Split text into sentence-like spans, returned as (start, end) char offsets."""
    spans, start = [], 0
    for m in _SENTENCE_END_RE.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _split_long_span(text: str, start: int, end: int, max_words: int) -> list[tuple[int, int]]:
    """Split a single over-long sentence on word boundaries."""
    words = list(_WORD_RE.finditer(text, start, end))
    out = []
    for i in range(0, len(words), max_words):
        group = words[i : i + max_words]
        out.append((group[0].start(), group[-1].end()))
    return out


def chunk_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[tuple[int, int]]:
    """Group sentences into chunks of at most `max_words` words, keeping char offsets."""
    chunks: list[tuple[int, int]] = []
    cur_start, cur_end, cur_words = None, None, 0
    for s, e in _split_spans(text):
        n = _word_count(text[s:e])
        pieces = _split_long_span(text, s, e, max_words) if n > max_words else [(s, e)]
        for ps, pe in pieces:
            pn = _word_count(text[ps:pe])
            if cur_start is not None and cur_words + pn > max_words:
                chunks.append((cur_start, cur_end))
                cur_start, cur_words = None, 0
            if cur_start is None:
                cur_start = ps
            cur_end = pe
            cur_words += pn
    if cur_start is not None:
        chunks.append((cur_start, cur_end))
    return chunks


def load_model(model_path: str | Path = DEFAULT_MODEL) -> GLiNER:
    model = GLiNER.from_pretrained(str(model_path), load_tokenizer=True)
    model.eval()
    return model


def extract_entities(
    model: GLiNER,
    text: str,
    labels: list[str] = DEFAULT_LABELS,
    threshold: float = 0.5,
) -> list[dict]:
    """Return entities as dicts with text, label, score, start, end (offsets in `text`)."""
    entities = []
    for c_start, c_end in chunk_text(text):
        chunk = text[c_start:c_end]
        for ent in model.predict_entities(chunk, labels, threshold=threshold, flat_ner=True):
            entities.append(
                {
                    "text": ent["text"],
                    "label": ent["label"],
                    "score": round(float(ent["score"]), 3),
                    "start": c_start + ent["start"],
                    "end": c_start + ent["end"],
                }
            )
    return sorted(entities, key=lambda e: e["start"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to a UTF-8 text report, or '-' for stdin")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, help="Entity types to extract")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum confidence (0-1)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Local model dir or HF model id")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of a table")
    args = parser.parse_args()

    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")

    t0 = time.perf_counter()
    model = load_model(args.model)
    t1 = time.perf_counter()
    entities = extract_entities(model, text, args.labels, args.threshold)
    t2 = time.perf_counter()

    if args.json:
        print(json.dumps(entities, ensure_ascii=False, indent=2))
    else:
        width = max((len(e["text"]) for e in entities), default=10)
        for e in entities:
            print(f"{e['text']:<{width}}  {e['label']:<18} {e['score']:.2f}  [{e['start']}:{e['end']}]")
        print(f"\n{len(entities)} entities | model load {t1 - t0:.1f}s | inference {t2 - t1:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
