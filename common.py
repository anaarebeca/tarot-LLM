import json

import pandas as pd

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