import argparse, csv, json
from pathlib import Path
from chunkers import slug
from common import load_cards, load_combos, notable_pairs
import retrieval

HDR = 600


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

    cards, combos = load_cards(), load_combos()
    golden = [json.loads(l) for l in open(a.golden, encoding="utf-8") if l.strip()]
    golden = [ex for ex in golden if ex.get("split") == a.split]

    ranks = {}

    def run(method, kind, items, hit):
        ranks.setdefault((method, kind), []).append(rank_of(items, hit))

    for ex in golden:
        drawn = [c for c in (ex["input"].get("cards") or []) if c["card"] in cards]
        for c in drawn:
            q, hit = f'{c["card"]} {c["orientation"]}', card_hit(a.strategy, c["card"], c["orientation"])
            run("naive", "card", retrieval.naive(a.strategy, a.model, q, a.k), hit)
            if a.strategy == "structure":
                run("filtered", "card", retrieval.metadata_filtered(
                    a.strategy, a.model, q, a.k, card=c["card"], orientation=c["orientation"]), hit)
        names = [c["card"] for c in drawn]
        for x, y in notable_pairs(names, combos):
            q, hit = f"{x} and {y} together", combo_hit(a.strategy, combos[frozenset((x, y))], x, y)
            run("naive", "combination", retrieval.naive(a.strategy, a.model, q, a.k), hit)
            if a.strategy == "structure":
                run("filtered", "combination",
                    retrieval.metadata_filtered(a.strategy, a.model, q, a.k, type="combination"), hit)
        if any(c["orientation"] == "reversed" for c in drawn):
            q, hit = "how to read reversed cards", guide_hit(a.strategy)
            run("naive", "guide", retrieval.naive(a.strategy, a.model, q, a.k), hit)
            if a.strategy == "structure":
                run("filtered", "guide",
                    retrieval.metadata_filtered(a.strategy, a.model, q, a.k, type="reversal"), hit)

    out = Path("results"); out.mkdir(exist_ok=True)
    path = out / f"retrieval_{a.split}.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["strategy", "model", "method", "kind", "n", "hit@1", "hit@3", "hit@5", "mrr"])
        for (method, kind), rs in sorted(ranks.items()):
            n = len(rs)
            hit_k = lambda k: round(sum(r is not None and r <= k for r in rs) / n, 3)
            mrr = round(sum(1 / r for r in rs if r) / n, 3)
            row = [a.strategy, a.model, method, kind, n, hit_k(1), hit_k(3), hit_k(5), mrr]
            w.writerow(row)
            print(row)


if __name__ == "__main__":
    main()