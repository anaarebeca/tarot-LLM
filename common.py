import json
from pathlib import Path

import pandas as pd

ORIENTATIONS = ("upright", "reversed")


def pair_key(a, b):
    return frozenset((a, b))

def load_combos(path="data/combinations.json"):
    import json
    combos = {}
    for i, item in enumerate(json.load(open(path, encoding="utf-8"))):
        key = pair_key(*item["cards"])
        assert key not in combos, f"Duplicate pair (any order): {item['cards']}"
        combos[key] = i
    return combos

def notable_pairs(drawn_cards, combos):
    from itertools import combinations
    return [(a, b) for a, b in combinations(drawn_cards, 2) if pair_key(a, b) in combos]

def load_cards(path="data/cards.csv"):
    return {r["name"]: r for _, r in pd.read_csv(path).iterrows()}

def load_spread(spread_id):
    return json.load(open(f"data/spreads/{spread_id}.json", encoding="utf-8"))


#added in week 8
def load_spreads(folder="data/spreads"):
    """spread id -> spread dict (all files in data/spreads)"""
    spreads = {}
    for path in sorted(Path(folder).glob("*.json")):
        spread = json.load(open(path, encoding="utf-8"))
        assert spread["num_cards"] == len(spread["positions"]), f"{path.name}: num_cards != number of positions"
        spreads[spread["id"]] = spread
    return spreads

def card_facts(row):
    """Normalized arcana/suit/element of one row from load_cards (Major Arcana have no suit, so it's '')."""
    arcana = "major" if "major" in str(row["arcana"]).lower() else "minor"
    suit = "" if arcana == "major" or pd.isna(row["suit"]) else str(row["suit"]).lower()
    element = "" if pd.isna(row["element"]) else str(row["element"]).lower()
    return {"arcana": arcana, "suit": suit, "element": element}