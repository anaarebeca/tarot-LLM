"""Builds golden/golden.jsonl (100 examples, 50/50 dev/test split).

  python generate_golden.py

- 60 examples are scripted (single, reversed, position, multi, personalized).
- 40 come from golden/manual.jsonl (10 personalized, 20 follow-up, 10 missing-input), see golden/README.md.
- Card keywords come from golden/card_keywords.csv. It is created on the first run (first 4 keywords per card
  and orientation). Skim it, fix ugly rows, rerun: the generator always reads that csv.
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from analysis import analyze_spread, salient_notes
from common import ORIENTATIONS, load_cards, load_combos, load_spreads, notable_pairs

SEED = 42
GOLDEN = Path("golden")
KEYWORDS_CSV = GOLDEN / "card_keywords.csv"
MANUAL = GOLDEN / "manual.jsonl"
OUT = GOLDEN / "golden.jsonl"

ALL_TOOLS = ["list_spreads", "get_spread", "update_user_context", "validate_reading", "get_card_meaning",
             "search_guides", "analyze_spread", "save_reading", "load_reading"]
ROUTES = {"reading": ["load_reading"],
          "card": ["load_reading", "get_card_meaning"],
          "guides": ["load_reading", "search_guides"]}
CONTEXTS = ["going through a divorce", "thinking about changing careers", "stressed about my exams",
            "moving to a new city next month", "starting my own business", "in a long-distance relationship",
            "burned out at work", "deciding whether to go back to school", "having a conflict with a close friend",
            "planning to buy my first home"]
TARGET_COUNTS = {"single": 10, "reversed": 10, "position": 10, "multi": 20, "personalized": 20, "followup": 20,
                 "missing": 10}

cards_info = load_cards()
spreads = load_spreads()
combos = load_combos()
names = list(cards_info)
PAIRS = [item["cards"] for item in json.load(open("data/combinations.json", encoding="utf-8"))]


#keywords
def ensure_keywords_csv():
    if KEYWORDS_CSV.exists():
        return
    GOLDEN.mkdir(exist_ok=True)
    with open(KEYWORDS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["card", "orientation", "keywords"])
        for name, row in cards_info.items():
            for o in ORIENTATIONS:
                raw = str(row[f"keywords_{o}"])
                kws = [k.strip().lower() for k in raw.replace(",", ";").split(";") if k.strip()]
                w.writerow([name, o, "; ".join(kws[:4])])
    print(f"Created {KEYWORDS_CSV}: skim it, fix ugly rows, then rerun.")


def load_keywords():
    ensure_keywords_csv()
    with open(KEYWORDS_CSV, newline="", encoding="utf-8") as f:
        return {(r["card"], r["orientation"]): [k.strip() for k in r["keywords"].split(";") if k.strip()]
                for r in csv.DictReader(f)}


keywords = load_keywords()


#helpers
def describe(spread_id, cards):
    parts = ", ".join(f"{c['position']}: {c['card']} ({c['orientation']})" for c in cards)
    return f"{spreads[spread_id]['name']} - {parts}"


def tools_for_reading(n_cards, has_pair, has_context, spread_named):
    required = ["get_spread", "validate_reading", "get_card_meaning", "save_reading"]
    optional = []
    (required if n_cards >= 2 else optional).append("analyze_spread")
    (required if has_pair else optional).append("search_guides")
    if has_context:
        required.append("update_user_context")
    if not spread_named:
        required.append("list_spreads")
    deps = [["validate_reading", "analyze_spread"], ["validate_reading", "save_reading"]]
    if not spread_named:
        deps.append(["list_spreads", "get_spread"])
    return required, optional, deps


def expected_block(required, optional, deps, spread_id):
    return {"required_tools": required, "optional_tools": optional,
            "unnecessary_tools": [t for t in ALL_TOOLS if t not in required and t not in optional],
            "dependencies": deps, "spread": spread_id}


def build_reading_example(ex_type, spread_id, cards, user_message, user_context=None, spread_named=True,
                          focus_position=None, extra_points=None):
    """Shared by the scripted examples and the manual 'personalized' rows."""
    shown = [c for c in cards if focus_position is None or c["position"] == focus_position]
    card_points = {f"{c['card']}|{c['orientation']}|{c['position']}": keywords[(c["card"], c["orientation"])]
                   for c in shown}
    result = analyze_spread(cards, cards_info, combos)
    whole_reading = focus_position is None and len(cards) > 1
    has_pair = bool(result["notable_pairs"]) and whole_reading
    required, optional, deps = tools_for_reading(len(cards), has_pair, bool(user_context), spread_named)
    return {
        "type": ex_type,
        "input": {"user_message": user_message, "spread": spread_id if spread_named else None, "cards": cards,
                  "user_context": user_context, "previous_reading": None},
        "card_points": card_points,
        "connection_points": salient_notes(result) if whole_reading else [],
        "context_points": [f"refers to the user's situation: {user_context}"] if user_context else [],
        "answer_points": extra_points or [],
        "expected": expected_block(required, optional, deps, spread_id),
    }


def draw(rng, spread_id, want_pair=None, orientations=None):
    """Random cards for a spread. Without want_pair: no notable pair. With it: exactly that pair."""
    positions = spreads[spread_id]["positions"]
    n = len(positions)
    for _ in range(2000):
        if want_pair:
            others = rng.sample([c for c in names if c not in want_pair], n - 2)
            chosen = others + list(want_pair)
            rng.shuffle(chosen)
            ok = len(notable_pairs(chosen, combos)) == 1
        else:
            chosen = rng.sample(names, n)
            ok = not notable_pairs(chosen, combos)
        if ok:
            break
    else:
        raise RuntimeError("could not draw a clean spread, check combinations.json")
    ors = orientations or [rng.choice(ORIENTATIONS) for _ in chosen]
    return [{"position": p["position"], "card": c, "orientation": o} for p, c, o in zip(positions, chosen, ors)]


# scripted examples
def scripted(rng):
    out = []

    for i in range(10):  # single upright, half without naming the spread (tests list_spreads)
        cards = draw(rng, "one-card", orientations=["upright"])
        named = i % 2 == 0
        c = cards[0]
        msg = (f"One-card reading: I drew {c['card']} (upright). What does it mean?" if named
               else f"I drew {c['card']} (upright). What does it mean for me?")
        out.append(build_reading_example("single", "one-card", cards, msg, spread_named=named))

    for i in range(10):  # single reversed
        cards = draw(rng, "one-card", orientations=["reversed"])
        named = i % 2 == 0
        c = cards[0]
        msg = (f"One-card reading: I drew {c['card']} (reversed). What does it mean?" if named
               else f"I drew {c['card']} reversed. What should I take from it?")
        out.append(build_reading_example("reversed", "one-card", cards, msg, spread_named=named))

    for _ in range(10):  # position-specific: question about one position of a three-card spread
        cards = draw(rng, "three-card")
        focus = rng.choice(cards)
        msg = (f"{describe('three-card', cards)}. "
               f"What does {focus['card']} ({focus['orientation']}) mean in the {focus['position']} position?")
        out.append(build_reading_example("position", "three-card", cards, msg, focus_position=focus["position"]))

    pair_list = rng.sample(PAIRS, 7)  # 7 of the 20 multi readings contain a notable pair
    specs = [("three-card", i < 5) for i in range(14)] + [("celtic-cross", i < 2) for i in range(6)]
    rng.shuffle(specs)
    pair_for = {"three-card": iter(pair_list[:5]), "celtic-cross": iter(pair_list[5:])}
    for spread_id, with_pair in specs:
        want = tuple(next(pair_for[spread_id])) if with_pair else None
        cards = draw(rng, spread_id, want_pair=want)
        msg = f"{describe(spread_id, cards)}. Can you read this spread for me?"
        out.append(build_reading_example("multi", spread_id, cards, msg))

    for ctx in CONTEXTS:  # personalized, scripted half
        cards = draw(rng, "three-card")
        msg = f"{describe('three-card', cards)}. I'm {ctx}. What should I know?"
        out.append(build_reading_example("personalized", "three-card", cards, msg, user_context=ctx))
    return out


#manual examples
def require(row, fields, n):
    for f in fields:
        if f not in row:
            raise ValueError(f"manual.jsonl line {n}: missing field '{f}'")


def check_cards(cards, n):
    for c in cards:
        if c["card"] not in cards_info:
            raise ValueError(f"manual.jsonl line {n}: unknown card '{c['card']}' (run check_data.py)")
        if c["orientation"] not in ORIENTATIONS:
            raise ValueError(f"manual.jsonl line {n}: bad orientation '{c['orientation']}'")


def manual_rows():
    if not MANUAL.exists():
        return []
    out = []
    for n, line in enumerate(MANUAL.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        t = row.get("type")
        if t == "personalized":
            require(row, ["user_message", "spread", "cards", "user_context"], n)
            check_cards(row["cards"], n)
            out.append(build_reading_example("personalized", row["spread"], row["cards"], row["user_message"],
                                             user_context=row["user_context"], extra_points=row.get("answer_points")))
        elif t == "followup":
            require(row, ["user_message", "previous_reading", "route", "answer_points"], n)
            prev = row["previous_reading"]
            check_cards(prev["cards"], n)
            required = ROUTES[row["route"]]
            targets = []  # what retrieval should find for this follow-up (used by eval_retrieval.py)
            if row.get("focus_card"):
                fc = row["focus_card"]
                targets.append({"kind": "card", "card": fc["card"], "orientation": fc["orientation"]})
            if row.get("focus_pair"):
                targets.append({"kind": "combination", "cards": row["focus_pair"]})
            out.append({"type": "followup",
                        "input": {"user_message": row["user_message"], "spread": prev["spread"], "cards": None,
                                  "user_context": prev.get("user_context"), "previous_reading": prev},
                        "card_points": {}, "connection_points": [], "context_points": [],
                        "answer_points": row["answer_points"],
                        "expected": expected_block(required, row.get("optional_tools", []),
                                                   [[required[0], t2] for t2 in required[1:]], prev["spread"]),
                        "retrieval": targets})
        elif t == "missing":
            require(row, ["user_message", "kind", "answer_points"], n)
            required = ["list_spreads"] if row["kind"] == "no_spread" else ["get_spread", "validate_reading"]
            deps = [["get_spread", "validate_reading"]] if row["kind"] == "invalid" else []
            out.append({"type": "missing",
                        "input": {"user_message": row["user_message"], "spread": row.get("spread"),
                                  "cards": row.get("cards"), "user_context": None, "previous_reading": None},
                        "card_points": {}, "connection_points": [], "context_points": [],
                        "answer_points": row["answer_points"],
                        "expected": expected_block(required, [], deps, row.get("spread"))})
        else:
            raise ValueError(f"manual.jsonl line {n}: type must be personalized, followup or missing")
    return out


#split + write
def main():
    if len(PAIRS) < 7:
        raise SystemExit("data/combinations.json needs at least 7 pairs for the golden set")
    rng = random.Random(SEED)
    examples = scripted(rng) + manual_rows()
    by_type = defaultdict(list)
    for ex in examples:
        by_type[ex["type"]].append(ex)
    for group in by_type.values():  # stratified 50/50 split, fixed seed
        rng.shuffle(group)
        for i, ex in enumerate(group):
            ex["split"] = "dev" if i < (len(group) + 1) // 2 else "test"
    counters = Counter()
    final = []
    for t in TARGET_COUNTS:
        for ex in by_type.get(t, []):
            counters[t] += 1
            final.append({"id": f"{t}_{counters[t]:03d}", "type": t, "split": ex["split"],
                          **{k: v for k, v in ex.items() if k not in ("type", "split")}})
    GOLDEN.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for ex in final:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final)} examples to {OUT}\n")
    print(f"{'type':14}{'have':>6}{'target':>8}{'dev':>6}{'test':>6}")
    for t, target in TARGET_COUNTS.items():
        group = by_type.get(t, [])
        print(f"{t:14}{len(group):>6}{target:>8}{sum(e['split'] == 'dev' for e in group):>6}"
              f"{sum(e['split'] == 'test' for e in group):>6}")
    pairs = sum(any(p.startswith("notable pair") for p in e["connection_points"]) for e in by_type["multi"])
    print(f"\nmulti readings with a notable pair: {pairs} of {len(by_type['multi'])}")
    if len(final) < 100:
        print("Manual examples still missing: write them in golden/manual.jsonl (see golden/README.md)")


if __name__ == "__main__":
    main()