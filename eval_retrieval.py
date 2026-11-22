"""Retrieval check against the golden dataset.

  python eval_retrieval.py --strategy structure --model minilm
  python eval_retrieval.py --strategy fixed --model bge
Run it once per collection. Results go to results/retrieval_<split>.csv; rerunning a collection replaces its rows.
The default split is dev. Use --split test only for the final run (week 13).
"""
import argparse, csv, json
from pathlib import Path
from chunkers import slug
from common import load_cards, load_combos, notable_pairs
import retrieval

HDR = 600
HEADER = ["strategy", "model", "method", "kind", "n", "hit@1", "hit@3", "hit@5", "mrr"]


def card_hit(strategy, card, orient):
    cid, header = f"card::{slug(card)}::{orient}", f"{card} - {orient}"
    return (lambda it: it[0] == cid) if strategy == "structure" else (lambda it: header in it[1][:HDR])


def combo_hit(strategy, idx, a, b):
    if strategy == "structure":
        return lambda it: it[0].startswith(f"combo::{idx}::")
    heads = (f"Combination: {a} + {b}", f"Combination: {b} + {a}")
    return lambda it: any(h in it[1][:HDR] for h in heads)


def guide_hit(strategy):
    if strategy == "structure":
        return lambda it: it[0].startswith("guide::reversals::")
    return lambda it: it[2]["source"] == "reversals.md"


def rank_of(items, is_hit):
    for i, it in enumerate(items, 1):
        if is_hit(it):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["structure", "fixed"], default="structure")
    ap.add_argument("--model", choices=list(retrieval.MODELS), default="minilm")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--golden", default="golden/golden.jsonl")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    if a.split == "test":
        print("WARNING: the test split is for the final run (week 13). Don't tune on it.")

    cards, combos = load_cards(), load_combos()
    golden = [json.loads(l) for l in open(a.golden, encoding="utf-8") if l.strip()]
    golden = [ex for ex in golden if ex.get("split") == a.split]

    ranks = {}

    def run(method, kind, items, hit):
        ranks.setdefault((method, kind), []).append(rank_of(items, hit))

    def check_card(card, orient):
        q, hit = f"{card} {orient}", card_hit(a.strategy, card, orient)
        run("naive", "card", retrieval.naive(a.strategy, a.model, q, a.k), hit)
        if a.strategy == "structure":
            run("filtered", "card", retrieval.metadata_filtered(
                a.strategy, a.model, q, a.k, card=card, orientation=orient), hit)

    def check_pair(x, y):
        q, hit = f"{x} and {y} together", combo_hit(a.strategy, combos[frozenset((x, y))], x, y)
        run("naive", "combination", retrieval.naive(a.strategy, a.model, q, a.k), hit)
        if a.strategy == "structure":
            run("filtered", "combination",
                retrieval.metadata_filtered(a.strategy, a.model, q, a.k, type="combination"), hit)

    for ex in golden:
        if ex["type"] == "missing":  # these examples have no valid reading to retrieve for
            continue
        drawn = [c for c in (ex["input"].get("cards") or []) if c["card"] in cards]
        for c in drawn:
            check_card(c["card"], c["orientation"])
        for x, y in notable_pairs([c["card"] for c in drawn], combos):
            check_pair(x, y)
        if any(c["orientation"] == "reversed" for c in drawn):
            q, hit = "how to read reversed cards", guide_hit(a.strategy)
            run("naive", "guide", retrieval.naive(a.strategy, a.model, q, a.k), hit)
            if a.strategy == "structure":
                run("filtered", "guide",
                    retrieval.metadata_filtered(a.strategy, a.model, q, a.k, type="reversal"), hit)
        for t in ex.get("retrieval", []):  # follow-ups: the card or pair the question is about
            if t["kind"] == "card":
                check_card(t["card"], t["orientation"])
            elif frozenset(t["cards"]) in combos:
                check_pair(*t["cards"])

    new_rows = []
    for (method, kind), rs in sorted(ranks.items()):
        n = len(rs)
        hit_k = lambda k: round(sum(r is not None and r <= k for r in rs) / n, 3)
        mrr = round(sum(1 / r for r in rs if r) / n, 3)
        new_rows.append([a.strategy, a.model, method, kind, n, hit_k(1), hit_k(3), hit_k(5), mrr])
        print(new_rows[-1])

    out = Path("results"); out.mkdir(exist_ok=True)
    path = out / f"retrieval_{a.split}.csv"
    kept = []
    if path.exists():  # drop the old rows of this collection, so reruns don't pile up duplicates
        with open(path, newline="", encoding="utf-8") as f:
            kept = [r for r in list(csv.reader(f))[1:] if r[:2] != [a.strategy, a.model]]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(kept + new_rows)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()