"""The pure-code logic of the analyze_spread tool.

The golden-dataset generator and the MCP server (week 9) both import this,
so the golden "connection points" can never drift from what the tool returns.
"""
from collections import Counter

from common import card_facts, notable_pairs


def _dominant(values, min_count=2):
    """Most common non-empty value, only if it appears >= min_count times and is not tied."""
    top = Counter(v for v in values if v).most_common(2)
    if not top or top[0][1] < min_count:
        return None
    if len(top) > 1 and top[1][1] == top[0][1]:
        return None
    return top[0]  # (value, count)


def analyze_spread(cards, card_info, combos):
    """cards: [{position, card, orientation}], card_info: load_cards(), combos: load_combos()."""
    names = [c["card"] for c in cards]
    n = len(cards)
    facts = [card_facts(card_info[x]) for x in names]
    reversed_n = sum(c["orientation"] == "reversed" for c in cards)
    return {
        "num_cards": n,
        "dominant_suit": _dominant([f["suit"] for f in facts]),      # (suit, count) or None
        "dominant_element": _dominant([f["element"] for f in facts]),
        "major_count": sum(f["arcana"] == "major" for f in facts),
        "reversed_count": reversed_n,
        "reversed_ratio": reversed_n / n if n else 0.0,
        "notable_pairs": notable_pairs(names, combos),
    }


def connection_notes(result):
    """All notes, as handed to the SLM in the prompt."""
    n = result["num_cards"]
    notes = [f"notable pair: {a} + {b}" for a, b in result["notable_pairs"]]
    if result["dominant_suit"]:
        s, c = result["dominant_suit"]
        notes.append(f"dominant suit: {s.capitalize()} ({c} of {n} cards)")
    if result["dominant_element"]:
        e, c = result["dominant_element"]
        notes.append(f"dominant element: {e.capitalize()} ({c} of {n} cards)")
    notes.append(f"Major Arcana: {result['major_count']} of {n} cards")
    notes.append(f"reversed cards: {result['reversed_count']} of {n} ({round(result['reversed_ratio'] * 100)}%)")
    return notes


def salient_notes(result):
    """Only the connections a reading must mention (these become the golden 'connection points').
    Kept short on purpose: every extra keypoint makes the LLM judge's job noisier."""
    n = result["num_cards"]
    notes = [f"notable pair: {a} + {b}" for a, b in result["notable_pairs"]]
    if n >= 3:
        if result["dominant_suit"]:
            s, c = result["dominant_suit"]
            notes.append(f"dominant suit: {s.capitalize()} ({c} of {n} cards)")
        if result["major_count"] * 2 >= n:
            notes.append(f"Major Arcana: {result['major_count']} of {n} cards")
        if result["reversed_count"] * 2 > n:
            notes.append(f"reversed cards: {result['reversed_count']} of {n} (most of the spread)")
    return notes