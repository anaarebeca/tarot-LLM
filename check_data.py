import json, pandas as pd
names = set(pd.read_csv("data/cards.csv")["name"])
bad = [c for it in json.load(open("data/combinations.json", encoding="utf-8"))
       for c in it["cards"] if c not in names]
print("OK" if not bad else f"Unknown card names: {bad}")