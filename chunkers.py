"""
Two ways of cutting the tarot data into chunks.
-structure_chunks: one chunk per card orientation, one per combination text,
   guide texts split section by section (our main strategy).
-fixed_chunks: fixed-size token windows over the same text (baseline).
"""
import json
import re
from pathlib import Path

import pandas as pd

SUIT_ELEMENT = {"wands": "fire", "cups": "water", "swords": "air", "pentacles": "earth"}

CARD_COLS = {"name": "name", "arcana": "arcana", "suit": "suit", "element": "element"}
ORIENT_COLS = {
    "upright": {
        "keywords": "keywords_upright",
        "general": "meaning_upright",
        "love": "love",
        "career": "career",
        "mood": "mood",
        "spiritual": "spiritual",
    },
    "reversed": {
        "keywords": "keywords_reversed",
        "general": "meaning_reversed",
        "love": "love_reversed",
        "career": "career_reversed",
        "mood": "mood_reversed",
        "spiritual": "spiritual_reversed",
    },
}

GUIDE_TYPES = {"suits": "suit", "reversals": "reversal"}  # file name -> chunk type


def meta(**kw):
    """Every chunk carries the same metadata keys (Chroma doesn't accept None)."""
    base = {"source": "", "type": "", "card": "", "cards": "", "suit": "",
            "element": "", "arcana": "", "orientation": ""}
    base.update(kw)
    return base


def val(row, col):
    if not col or col not in row.index or pd.isna(row[col]):
        return ""
    return str(row[col]).strip()


def card_chunks(csv_path):
    df = pd.read_csv(csv_path)
    chunks = []
    for _, row in df.iterrows():
        name = val(row, CARD_COLS["name"])
        arcana = "major" if "major" in val(row, CARD_COLS["arcana"]).lower() else "minor"
        suit = "" if arcana == "major" else val(row, CARD_COLS["suit"]).lower()
        element = val(row, CARD_COLS["element"]).lower() or SUIT_ELEMENT.get(suit, "")
        for orientation, cols in ORIENT_COLS.items():
            lines = [f"{name} - {orientation}", f"Arcana: {arcana}" + (f", suit: {suit}" if suit else "")]
            filled = 0
            for field, col in cols.items():
                text = val(row, col)
                if text:
                    lines.append(f"{field.capitalize()}: {text}")
                    filled += 1
            if filled == 0:
                print(f"WARNING: no text found for {name} ({orientation}), check ORIENT_COLS")
                continue
            chunks.append({
                "id": f"card::{slug(name)}::{orientation}",
                "text": "\n".join(lines),
                "metadata": meta(source=Path(csv_path).name, type="card", card=name, suit=suit,
                                 element=element, arcana=arcana, orientation=orientation),
            })
    return chunks


def combination_chunks(path):
    path = Path(path)
    if not path.exists():
        return []
    chunks = []
    for i, item in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        a, b = item["cards"]
        chunks.append({
            "id": f"combo::{i}::{slug(a)}::{slug(b)}",
            "text": f"Combination: {a} + {b}\n{item['text']}",
            "metadata": meta(source=path.name, type="combination", cards="|".join(sorted([a,b]))),
        })
    return chunks


def guide_chunks(guides_dir):
    chunks = []
    for path in sorted(Path(guides_dir).glob("*.md")):
        gtype = GUIDE_TYPES.get(path.stem, path.stem)
        sections = re.split(r"(?m)^## ", path.read_text(encoding="utf-8"))[1:]
        for i, sec in enumerate(sections):
            heading = sec.split("\n", 1)[0].strip()
            suit = heading.lower() if heading.lower() in SUIT_ELEMENT else ""
            chunks.append({
                "id": f"guide::{path.stem}::{i}",
                "text": sec.strip(),
                "metadata": meta(source=path.name, type=gtype, suit=suit, element=SUIT_ELEMENT.get(suit, "")),
            })
    return chunks


def structure_chunks(data_dir):
    data_dir = Path(data_dir)
    return (card_chunks(data_dir / "cards.csv")
            + combination_chunks(data_dir / "combinations.json")
            + guide_chunks(data_dir / "guides"))


def fixed_chunks(data_dir, tokenizer, size=300, overlap=50):
    """Baseline: slide a window of `size` tokens (step size-overlap) over each source's full text.
    Chunks ignore card boundaries on purpose, that's what we compare against."""
    data_dir = Path(data_dir)
    old_max = tokenizer.model_max_length
    tokenizer.model_max_length = 10 ** 9  # we only count tokens here
    streams = {"cards.csv": "\n\n".join(c["text"] for c in card_chunks(data_dir / "cards.csv"))}
    combos = combination_chunks(data_dir / "combinations.json")
    if combos:
        streams["combinations.json"] = "\n\n".join(c["text"] for c in combos)
    for path in sorted((data_dir / "guides").glob("*.md")):
        streams[path.name] = path.read_text(encoding="utf-8")

    chunks = []
    step = size - overlap
    for source, text in streams.items():
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        start, n = 0, 0
        while start < len(offsets):
            window = offsets[start:start + size]
            chunks.append({
                "id": f"fixed::{slug(source)}::{n}",
                "text": text[window[0][0]:window[-1][1]],
                "metadata": meta(source=source, type="fixed"),
            })
            if start + size >= len(offsets):
                break
            start += step
            n += 1
    tokenizer.model_max_length = old_max  # restore, the model shares this tokenizer
    return chunks


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")