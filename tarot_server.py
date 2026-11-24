#run: python tarot_server.py
import difflib
import json
import os
import re
import uuid
from itertools import combinations
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

import retrieval
from analysis import analyze_spread as run_analysis, connection_notes
from common import ORIENTATIONS, load_cards, load_combos, load_spreads

STRATEGY = "structure"
MODEL = os.environ.get("TAROT_MODEL", "minilm")
THEMES = ("general", "love", "career", "mood", "spiritual")
GUIDE_TYPES = {"combination": "combination", "combinations": "combination", "suit": "suit",
               "suits": "suit", "reversal": "reversal", "reversals": "reversal"}
SESSIONS = Path("sessions")

CARDS = load_cards()
COMBOS = load_combos()
SPREADS = load_spreads()
NAMES_LOWER = {n.lower(): n for n in CARDS}

VALIDATED = set()
CONTEXT = {}
READINGS = {}
STATE = {"latest": None}

mcp = FastMCP("tarot")


class CardInput(BaseModel):
    position: int | str | None = None
    card: str | None = None
    orientation: str | None = None

def _json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1)


def _unknown_card(name):
    name = (name or "").strip()
    close = difflib.get_close_matches(name.lower(), list(NAMES_LOWER), n=1, cutoff=0.6)
    return f"Unknown card '{name}'" + (f", did you mean '{NAMES_LOWER[close[0]]}'?" if close else "")


def _normalize(spread_id, cards):

    spread = SPREADS.get(spread_id)
    if not spread:
        return None, [], [f"Unknown spread '{spread_id}'. Available spreads: {', '.join(SPREADS)}"]
    by_num = {p["position"]: p for p in spread["positions"]}
    by_name = {p["name"].lower(): p for p in spread["positions"]}
    valid_pos = ", ".join(f"{p['position']} ({p['name']})" for p in spread["positions"])
    problems, out, seen_cards, seen_pos = [], [], set(), set()
    if len(cards) != spread["num_cards"]:
        problems.append(f"Spread '{spread_id}' needs {spread['num_cards']} card(s), got {len(cards)}")
    for c in cards:
        raw_name = (c.get("card") or "").strip()
        canon = NAMES_LOWER.get(raw_name.lower())
        who = canon or raw_name or "a card"
        if not canon:
            problems.append(_unknown_card(raw_name))
        else:
            if canon in seen_cards:
                problems.append(f"Duplicate card: {canon}")
            seen_cards.add(canon)

        pos, raw_pos = None, c.get("position")
        if raw_pos is None or str(raw_pos).strip() == "":
            problems.append(f"{who}: position is missing")
        else:
            s = str(raw_pos).strip()
            pos = by_num.get(int(s)) if s.isdigit() else by_name.get(s.lower())
            if not pos:
                problems.append(f"{who}: unknown position '{raw_pos}'. Valid positions: {valid_pos}")
            else:
                if pos["position"] in seen_pos:
                    problems.append(f"Duplicate position: {pos['position']} ({pos['name']})")
                seen_pos.add(pos["position"])

        label = f"Position {pos['position']}" if pos else who
        o = (c.get("orientation") or "").strip().lower()
        if not o:
            problems.append(f"{label}: orientation is missing")
        elif o not in ORIENTATIONS:
            problems.append(f"{label}: '{o}' is not a valid orientation, use upright or reversed")
        if canon and pos and o in ORIENTATIONS:
            out.append({"position": pos["position"], "card": canon, "orientation": o})
    out.sort(key=lambda x: x["position"])
    return spread, out, problems


def _fingerprint(spread_id, canon):
    return (spread_id, tuple((c["position"], c["card"], c["orientation"]) for c in canon))


def _check(spread_id, cards):
    _, canon, problems = _normalize(spread_id, [c.model_dump() for c in cards])
    if problems or _fingerprint(spread_id, canon) not in VALIDATED:
        return canon, "Reading was not validated. Call validate_reading first."
    return canon, None


def _parse_fields(text):
    fields, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^(Keywords|General|Love|Career|Mood|Spiritual): ?(.*)$", line)
        if m:
            cur = m.group(1).lower()
            fields[cur] = m.group(2).strip()
        elif cur:
            fields[cur] += " " + line.strip()
    return fields

@mcp.tool()
def list_spreads() -> str:
    """
    Returns all available spreads with their id, name, number of cards, theme and a one-line description.
    Use this when the user has not named a spread, to choose the best fit (a one-card spread for a quick question, three cards for past/present/future, Celtic Cross for a complex situation).
    Do not use it if the user already named a spread.
    """
    return _json([{"id": s["id"], "name": s["name"], "num_cards": s["num_cards"], "theme": s["theme"],
                   "description": s["description"]} for s in SPREADS.values()])


@mcp.tool()
def get_spread(spread_id: str) -> str:
    """
    Loads one spread and returns its ordered positions, each with its meaning and theme (general, love, career, mood, spiritual).
    Use it once a spread is chosen, before looking at cards, because every card is interpreted through its position.
    Input: spread_id (e.g. 'three-card')
    """
    spread = SPREADS.get(spread_id)
    if not spread:
        return f"Unknown spread '{spread_id}'. Available spreads: {', '.join(SPREADS)}"
    return _json({"id": spread["id"], "name": spread["name"], "num_cards": spread["num_cards"],
                  "positions": spread["positions"]})


@mcp.tool()
def update_user_context(situation: str | None = None, topic: str | None = None,
                        question: str | None = None, mood: str | None = None) -> str:
    """
    Stores personal details the user shared (situation, topic, the question they want answered, mood) as structured fields, so the reading can be personalized.
    Use it whenever the user shares something personal, in the first message or later.
    Do not invent details the user did not give.
    Fields you pass are merged into what is already stored.
    """
    new = {"situation": situation, "topic": topic, "question": question, "mood": mood}
    new = {k: v for k, v in new.items() if v}
    if not new:
        return "Nothing to store: pass at least one of situation, topic, question, mood."
    CONTEXT.update(new)
    return _json({"stored_context": CONTEXT})


@mcp.tool()
def validate_reading(spread_id: str, cards: list[CardInput]) -> str:
    """
    Checks that the submitted cards fit the chosen spread: correct number of cards, no duplicate cards, valid card names, a valid position for every card, and an orientation (upright or reversed) for every card.
    Must be called, and must pass, before looking up meanings, analyzing or saving.
    If it fails, tell the user what to fix and do not continue.
    Input: spread_id and a list of {position, card, orientation}
    Returns 'ok' or a list of problems.
    """
    _, canon, problems = _normalize(spread_id, [c.model_dump() for c in cards])
    if problems:
        return "Reading is NOT valid:\n- " + "\n- ".join(problems) + \
               "\nTell the user what to fix and do not continue."
    VALIDATED.add(_fingerprint(spread_id, canon))
    return "ok"


@mcp.tool()
def get_card_meaning(card: str, orientation: str, theme: str = "general") -> str:
    """
    Returns the curated meaning of one card for one orientation: keywords, the general meaning, plus the field matching the theme of its position (love, career, mood or spiritual).
    Reversed cards return their own reversed meaning, never flip the upright one yourself.
    Call it for every drawn card (use the theme of its position, from get_spread), and also for follow-ups about one card's detail.
    Input: card name, orientation ('upright' or 'reversed'), theme (general, love, career, mood, spiritual)
    """
    canon = NAMES_LOWER.get((card or "").strip().lower())
    if not canon:
        return _unknown_card(card)
    o = (orientation or "").strip().lower()
    if o not in ORIENTATIONS:
        return f"Invalid orientation '{orientation}'. Use 'upright' or 'reversed'."
    t = (theme or "general").strip().lower()
    if t not in THEMES:
        return f"Unknown theme '{theme}'. Use one of: {', '.join(THEMES)}."
    hits = retrieval.metadata_filtered(STRATEGY, MODEL, f"{canon} {o}", card=canon, orientation=o)
    if not hits:
        return f"No chunk found for {canon} ({o}). Has ingest.py been run for model '{MODEL}'?"
    f = _parse_fields(hits[0][1])
    out = {"card": canon, "orientation": o, "theme": t,
           "keywords": f.get("keywords", ""), "general_meaning": f.get("general", "")}
    if t != "general":
        out["theme_meaning"] = f.get(t) or f"No {t} text for this card."
    return _json(out)


@mcp.tool()
def search_guides(query: str, type: str | None = None, cards: list[str] | None = None, k: int = 3) -> str:
    """
    Searches the guide texts about card combinations, suits and reversals.
    Use it for knowledge beyond a single card: how two cards interact, what a suit means, how reversals work.
    Input: query (free text), optional type filter ('combination', 'suit' or 'reversal'), optional list of card names
    If you pass two or more drawn cards, the exact combination text for each pair that has one is returned.
    If no combination text exists for those cards, you are told so.
    """
    gtype = None
    if type:
        gtype = GUIDE_TYPES.get(type.strip().lower())
        if not gtype:
            return f"Unknown type '{type}'. Use combination, suit or reversal."
    results = []
    if cards:
        names = []
        for c in cards:
            canon = NAMES_LOWER.get((c or "").strip().lower())
            if not canon:
                return _unknown_card(c)
            names.append(canon)
        if len(set(names)) >= 2 and gtype in (None, "combination"):
            col = retrieval.get_collection(STRATEGY, MODEL)
            for a, b in combinations(sorted(set(names)), 2):
                if frozenset((a, b)) in COMBOS:
                    res = col.get(where={"cards": "|".join(sorted([a, b]))})
                    results += [{"id": i, "type": m["type"], "text": d}
                                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
            if results:
                return _json(results)
            if gtype == "combination":
                return "No combination text exists for these cards."
    where = {"type": gtype} if gtype else {"type": {"$in": ["combination", "suit", "reversal"]}}
    hits = retrieval.naive(STRATEGY, MODEL, query, max(1, min(k, 5)), where=where)
    return _json([{"id": i, "type": m["type"], "text": d} for i, d, m in hits])


@mcp.tool()
def analyze_spread(spread_id: str, cards: list[CardInput]) -> str:
    """
    Looks at all drawn cards together with fixed rules and returns connection notes: dominant suit and element, number of Major Arcana cards, ratio of reversed cards, and notable pairs (pairs that have a combination text).
    Use it after validate_reading so you have explicit connections to narrate in the reading.
    Input: the same spread_id and cards that passed validation
    """
    canon, err = _check(spread_id, cards)
    if err:
        return err
    result = run_analysis(canon, CARDS, COMBOS)
    return _json({"connection_notes": connection_notes(result),
                  "notable_pairs": [list(p) for p in result["notable_pairs"]]})


@mcp.tool()
def save_reading(spread_id: str, cards: list[CardInput], interpretation: str, summary: str,
                 connection_notes: list[str] | None = None) -> str:
    """
    Stores the full structured reading (spread, cards, positions, orientations, the user context stored so far, connection notes, interpretation, short summary) so follow-up questions can reuse it.
    Call it once the reading is written.
    The cards must have passed validate_reading.
    Returns the reading_id."""
    canon, err = _check(spread_id, cards)
    if err:
        return err
    rid = "r_" + uuid.uuid4().hex[:8]
    reading = {"reading_id": rid, "spread": spread_id, "cards": canon, "user_context": dict(CONTEXT),
               "connection_notes": connection_notes or [], "interpretation": interpretation, "summary": summary}
    READINGS[rid] = reading
    STATE["latest"] = rid
    SESSIONS.mkdir(exist_ok=True)
    (SESSIONS / f"{rid}.json").write_text(_json(reading), encoding="utf-8")
    return f"Saved. reading_id = {rid}"


@mcp.tool()
def load_reading(reading_id: str | None = None) -> str:
    """
    Restores a saved reading.
    Use it at the start of every follow-up question instead of relying on the chat history.
    Input: reading_id, or leave it empty for the latest reading of this session
    """
    rid = reading_id or STATE["latest"]
    if rid and rid in READINGS:
        return _json(READINGS[rid])
    if rid and (SESSIONS / f"{rid}.json").exists():
        return (SESSIONS / f"{rid}.json").read_text(encoding="utf-8")
    return "No saved reading found for this session. Ask the user for the cards and spread."


if __name__ == "__main__":
    mcp.run()