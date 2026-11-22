"""Checks the data files: card names in combinations.json and golden/manual.jsonl must exist in cards.csv.

  python check_data.py
Run it after every change to the combinations or the manual examples.
"""
import difflib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from common import ORIENTATIONS, load_spreads, pair_key

names = set(pd.read_csv("data/cards.csv")["name"])
spreads = load_spreads()
problems = []


def check_card(name, where):
    if name not in names:
        hint = difflib.get_close_matches(name, names, n=1)
        problems.append(f"{where}: unknown card '{name}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))


combos = json.load(open("data/combinations.json", encoding="utf-8"))
seen = Counter()
for i, item in enumerate(combos):
    a, b = item["cards"]
    check_card(a, f"combinations.json #{i}")
    check_card(b, f"combinations.json #{i}")
    if a == b:
        problems.append(f"combinations.json #{i}: same card twice ({a})")
    seen[pair_key(a, b)] += 1
for key, count in seen.items():
    if count > 1:
        problems.append(f"combinations.json: pair {sorted(key)} appears {count} times (order doesn't matter)")

manual = Path("golden/manual.jsonl")
if manual.exists():
    for n, line in enumerate(manual.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"manual.jsonl line {n}: invalid JSON ({e})")
            continue
        if row.get("spread") and row["spread"] not in spreads:
            problems.append(f"manual.jsonl line {n}: unknown spread '{row['spread']}' (have: {', '.join(spreads)})")
        card_lists = [row.get("cards") or []]
        if row.get("previous_reading"):
            card_lists.append(row["previous_reading"].get("cards") or [])
        if row.get("type") != "missing":  # missing-input rows may contain wrong names on purpose
            for card_list in card_lists:
                for c in card_list:
                    check_card(c["card"], f"manual.jsonl line {n}")
                    if c.get("orientation") not in ORIENTATIONS:
                        problems.append(f"manual.jsonl line {n}: orientation '{c.get('orientation')}' must be upright or reversed")
            for key in ("focus_card",):
                if row.get(key):
                    check_card(row[key]["card"], f"manual.jsonl line {n}")
            for card in row.get("focus_pair") or []:
                check_card(card, f"manual.jsonl line {n}")

print(f"{len(names)} cards, {len(combos)} combinations, {len(spreads)} spreads ({', '.join(spreads)})")
if problems:
    print("\n".join(problems))
    sys.exit(1)
print("OK")