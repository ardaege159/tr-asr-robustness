"""Rule-based scoring of an assistant's tool calls against a spec's answer key.

Outcomes, from best to worst:
  correct         every expected action was called with the right arguments
  took_first      self-correction case: the model used the value the customer retracted
  wrong_args      right tool, wrong argument value
  wrong_tool      an expected action is missing and a different action was called instead
  missed          an expected action is missing and nothing replaced it
  policy_breach   an action was taken that the spec says must not happen
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

# Lookups have no side effects, so calling one before the real action is fine.
READ_ONLY_TOOLS = {"siparis_sorgula", "rezervasyon_sorgula", "bakiye_sorgula"}

# Arguments that identify something and must match character for character.
ID_ARGS = {"siparis_no", "kart_son4", "pnr", "abone_no"}

# Free-text arguments compared by similarity, not equality.
FUZZY_ARGS = {"yeni_adres", "isyeri"}
FUZZY_THRESHOLD = 0.8

DATE_ARGS = {"tarih", "yeni_tarih"}

# Only arguments listed in a spec's answer key are checked, so a handoff's free-text
# reason is never compared (its expected args are {}).

_TR_UPPER = str.maketrans({"i": "İ", "ı": "I"})
_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def tr_lower(s: str) -> str:
    return s.translate(_TR_LOWER).lower()


def norm_id(value) -> str:
    """'K7M-2xa' -> 'K7M2XA'; dotted Turkish i folds to plain I so PNRs compare."""
    s = str(value).translate(_TR_UPPER).upper().replace("İ", "I")
    return re.sub(r"[^0-9A-ZÇĞÖŞÜ]", "", s)


def norm_text(value) -> str:
    s = tr_lower(str(value))
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def norm_date(value) -> str | None:
    """Accept 2026-11-20, 2026/11/20, 20.11.2026 and 20/11/2026; return ISO or None."""
    s = str(value).strip()
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
    else:
        m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
        if not m:
            return None
        d, mo, y = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def _as_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(" ", "")
    # Turkish style "2.450,50" and plain "2450.5" both appear in model output.
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def arg_matches(name: str, expected, actual) -> bool:
    if actual is None:
        return False
    if name in ID_ARGS:
        return norm_id(expected) == norm_id(actual)
    if name in DATE_ARGS:
        return norm_date(actual) is not None and norm_date(actual) == norm_date(expected)
    if name in FUZZY_ARGS:
        a, b = norm_text(expected), norm_text(actual)
        return difflib.SequenceMatcher(None, a, b).ratio() >= FUZZY_THRESHOLD
    exp_num = _as_number(expected)
    if exp_num is not None and not isinstance(expected, str):
        act_num = _as_number(actual)
        return act_num is not None and abs(exp_num - act_num) < 0.01
    return norm_text(expected) == norm_text(actual)


@dataclass
class Score:
    outcome: str
    detail: str = ""
    extra_calls: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome == "correct"


def _args_ok(expected_args: dict, actual_args: dict) -> tuple[bool, str]:
    for name, exp in expected_args.items():
        if not arg_matches(name, exp, actual_args.get(name)):
            return False, name
    return True, ""


def score(spec: dict, calls: list[dict], condition: str = "clean") -> Score:
    """Score one model response.

    spec:      a line of data/specs.jsonl
    calls:     [{"tool": name, "args": {...}}, ...] in the order the model made them
    condition: "clean", "asr", "selfcorr" or "selfcorr_asr"; only the selfcorr ones can yield took_first
    """
    expected = spec["expected"]
    expected_tools = {e["tool"] for e in expected}
    actions = [c for c in calls if c["tool"] not in READ_ONLY_TOOLS or c["tool"] in expected_tools]
    extra = [c["tool"] for c in actions if c["tool"] not in expected_tools]

    # Answer key says: take no action (e.g. return window has passed).
    if not expected:
        if actions:
            return Score("policy_breach", f"called {actions[0]['tool']}", extra)
        return Score("correct")

    correction = spec.get("correction") if condition.startswith("selfcorr") else None
    worst: Score | None = None
    for exp in expected:
        same_tool = [c for c in actions if c["tool"] == exp["tool"]]
        if not same_tool:
            result = Score("wrong_tool", f"expected {exp['tool']}", extra) if extra else Score("missed", f"expected {exp['tool']}")
        elif any(_args_ok(exp["args"], c.get("args") or {})[0] for c in same_tool):
            continue
        else:
            call = same_tool[-1]
            _, bad = _args_ok(exp["args"], call.get("args") or {})
            got = (call.get("args") or {}).get(bad)
            if correction and correction["arg"] == bad and arg_matches(bad, correction["wrong"], got):
                result = Score("took_first", f"{bad}={got!r}", extra)
            else:
                result = Score("wrong_args", f"{bad}: expected {exp['args'][bad]!r}, got {got!r}", extra)
        if worst is None or _RANK[result.outcome] > _RANK[worst.outcome]:
            worst = result

    if worst is not None:
        return worst
    if extra:
        # All expected actions happened, but something else was done too.
        return Score("policy_breach", f"also called {extra[0]}", extra)
    return Score("correct")


_RANK = {"correct": 0, "took_first": 1, "wrong_args": 2, "wrong_tool": 3, "missed": 4, "policy_breach": 5}
